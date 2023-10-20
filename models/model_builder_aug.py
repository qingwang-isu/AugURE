import torch
import torch.nn as nn
from random import sample

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    Cited: https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: temperature
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = True


        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.r % batch_size == 0  # for simplicity

        # dequeue and enqueue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0] if hasattr(x,'shape') else len(x)
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0] if hasattr(x_gather,'shape') else len(x_gather)

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0] if hasattr(x,'shape') else len(x)
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0] if hasattr(x_gather,'shape') else len(x_gather)

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, sen_q, sen_k=None, is_eval=False, cluster_result=None, index=None, print_index=0, sen_aug=None, must_links_l=None, must_links_r=None, margin_info=False):
        """
        Input:
            sen_q: a batch of query sentences
            sen_k: a batch of key sentences
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
            must_links: a batch of must links sentences
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            # if print_index<3:
            #     counter = 0
            #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            #         print('eval param_q: ',param_q.data)
            #         print('eval param_k: ',param_k.data)
            #         counter+=1
            #         if counter>10:
            #             break

            # print('senq: ',sen_q)
            k = self.encoder_k(sentence_data=sen_q)
            # print('cal: ',k)
            k = nn.functional.normalize(k, dim=1)
            # print('nor: ',k)
            return k
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # sen_k, idx_unshuffle = self._batch_shuffle_ddp(sen_k)

            k = self.encoder_k(sentence_data=sen_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute query features
        q = self.encoder_q(sentence_data=sen_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if margin_info:
            # need to return a triple as "logits"
            logits = []
            for ids, (a_input_ids, a_attention_mask, a_e1_pos, a_e2_pos, a_aug_pos, p_input_ids, p_attention_mask, p_e1_pos, p_e2_pos, p_aug_pos, p2_input_ids, p2_attention_mask, p2_e1_pos, p2_e2_pos, p2_aug_pos) in enumerate(zip(sen_q[0], sen_q[1], sen_q[3], sen_q[4], sen_q[5], sen_k[0], sen_k[1], sen_k[3], sen_k[4], sen_k[5], sen_aug[0], sen_aug[1], sen_aug[2], sen_aug[3], sen_aug[4])):
                a_ipt_ids = a_input_ids.unsqueeze(0)
                a_att_mask = a_attention_mask.unsqueeze(0)
                a_e1_pos = a_e1_pos.unsqueeze(0)
                a_e2_pos = a_e2_pos.unsqueeze(0)
                a_aug_pos = a_aug_pos.unsqueeze(0)

                a = self.encoder_q(input_ids=a_ipt_ids, attention_mask=a_att_mask, e1_pos=a_e1_pos, e2_pos=a_e2_pos, aug_pos=a_aug_pos)
                a = nn.functional.normalize(a, dim=1)

                p_ipt_ids = p_input_ids.unsqueeze(0)
                p_att_mask = p_attention_mask.unsqueeze(0)
                p_e1_pos = p_e1_pos.unsqueeze(0)
                p_e2_pos = p_e2_pos.unsqueeze(0)
                p_aug_pos = p_aug_pos.unsqueeze(0)

                p = self.encoder_q(input_ids=p_ipt_ids, attention_mask=p_att_mask, e1_pos=p_e1_pos, e2_pos=p_e2_pos, aug_pos=p_aug_pos)
                p = nn.functional.normalize(p, dim=1)

                p2_ipt_ids = p2_input_ids.unsqueeze(0)
                p2_att_mask = p2_attention_mask.unsqueeze(0)
                p2_e1_pos = p2_e1_pos.unsqueeze(0)
                p2_e2_pos = p2_e2_pos.unsqueeze(0)
                p2_aug_pos = p2_aug_pos.unsqueeze(0)

                p2 = self.encoder_q(input_ids=p2_ipt_ids, attention_mask=p2_att_mask, e1_pos=p2_e1_pos, e2_pos=p2_e2_pos, aug_pos=p2_aug_pos)
                p2 = nn.functional.normalize(p2, dim=1)

                # get a column of queue as negatives
                neg_sample = self.queue[:, ids:(ids+1)].clone().detach()
                logits.append([a, p, neg_sample])
                logits.append([a, p2, neg_sample])


        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # cross sentence positive pairs
        ml_triplet_lst = []
        if must_links_l is not None:
            for n_ml, (l_mlinput_ids, l_mlattention_mask, l_mle1_pos, l_mle2_pos, l_mlaug_pos, mlinput_ids, mlattention_mask, mle1_pos, mle2_pos, mlaug_pos) in enumerate(zip(must_links_l[0], must_links_l[1], must_links_l[2], must_links_l[3], must_links_l[4], must_links_r[0], must_links_r[1], must_links_r[2], must_links_r[3], must_links_r[4])):
                # 5x128
                for idx in range(len(mlinput_ids)):
                    # 128
                    # left
                    l_ipt_ids = l_mlinput_ids[idx].unsqueeze(0)
                    l_att_mask = l_mlattention_mask[idx].unsqueeze(0)
                    l_me1_pos = l_mle1_pos[idx].unsqueeze(0)
                    l_me2_pos = l_mle2_pos[idx].unsqueeze(0)
                    l_maug_pos = l_mlaug_pos[idx].unsqueeze(0)

                    l_ml = self.encoder_q(input_ids=l_ipt_ids, attention_mask=l_att_mask, e1_pos=l_me1_pos, e2_pos=l_me2_pos, aug_pos=l_maug_pos)
                    l_ml = nn.functional.normalize(l_ml, dim=1)

                    # right
                    ipt_ids = mlinput_ids[idx].unsqueeze(0)
                    att_mask = mlattention_mask[idx].unsqueeze(0)
                    me1_pos = mle1_pos[idx].unsqueeze(0)
                    me2_pos = mle2_pos[idx].unsqueeze(0)
                    maug_pos = mlaug_pos[idx].unsqueeze(0)

                    ml = self.encoder_q(input_ids=ipt_ids, attention_mask=att_mask, e1_pos=me1_pos, e2_pos=me2_pos, aug_pos=maug_pos)
                    ml = nn.functional.normalize(ml, dim=1)
                    #print("ml")
                    #print(ml.size())
                    #print(ml)

                    # get a column of queue as negatives
                    neg_sample = self.queue[:, idx:(idx+1)].clone().detach()
                    ml_triplet_lst.append([l_ml, ml, neg_sample])

        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (relation2cluster,prototypes,density) in enumerate(zip(cluster_result['relation2cluster'],cluster_result['centroids'],cluster_result['density'])):

                pos_proto_id = relation2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]    

                # sample negative prototypes
                all_proto_id = [i for i in range(relation2cluster.max())]
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id,self.r if self.r <len(neg_proto_id) else len(neg_proto_id)) #sample r negative prototypes
                neg_prototypes = prototypes[neg_proto_id]    

                proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())
                
                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
                logits_proto /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            if must_links_l is not None:
                return logits, labels, ml_triplet_lst, proto_logits, proto_labels
            else:
                return logits, labels, None, proto_logits, proto_labels

        else:
            if must_links_l is not None:
                return logits, labels, ml_triplet_lst, None, None
            else:
                return logits, labels, None, None, None


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
