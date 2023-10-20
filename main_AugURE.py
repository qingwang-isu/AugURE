import argparse
import builtins
import math
import random
import shutil
import warnings

from sklearn.cluster import AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN, KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import faiss

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import copy
import models.model_builder_aug
from models import bert_model
from transformers import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
import os
import time, json
from torch.utils.data import TensorDataset
import scorer
import re
import time
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='AugURE for relation representation learning')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset',default='data_sample_for_example')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10003', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--low-dim', default=768*3, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.02, type=float,
                    help='softmax temperature')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--pcl-r', default=10, type=int,  # 10  640  16384
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--margin', default=0.75, type=float,
                    help='triplet loss margin')
parser.add_argument('--num-cluster', default='4,6,10,16', type=str,
                    help='number of clusters, it is used in kmeans exemplar')
parser.add_argument('--warmup-epoch', default=0, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='tac_margin_rp_info', type=str,
                    help='experiment directory')
parser.add_argument('--max-length', default=128, type=int,
                    help='max length of sentence to be feed into bert (default 128)')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='adam_epsilon (default: 1e-8)')
parser.add_argument('--add-word-num', default=2, type=int,
                    help='data augmentation words number (default 3)')
parser.add_argument('--repeat-index', default=1, type=int,
                    help='repeat run index')
parser.add_argument('--alpha', default=10, type=int,
                    help='alpha of phi')
parser.add_argument('--attention', action='store_true',
                    help='use attention to update exampler')
parser.add_argument('--use-relation-span', action='store_true',
                    help='whether using relation span for data augmentation')

# ------------------------init parameters----------------------------

CUDA = "0,1,2,3,4"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

# Base code Cited: https://github.com/salesforce/PCL and https://github.com/thu-bpm/hiure
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.num_cluster = args.num_cluster.split(',')

    args.low_dim = (args.add_word_num+2)*768

    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
        os.mkdir(args.exp_dir + '/tsne/')
        os.mkdir(args.exp_dir + '/plot_data/')

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(args.dist_url)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    bert_encoder = bert_model.RelationClassification.from_pretrained(
        "../AugURE-master/bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.low_dim,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    tokenizer = get_tokenizer(args)
    bert_encoder.resize_token_embeddings(len(tokenizer))

    model = models.model_builder_aug.MoCo(
        bert_encoder,
        args.low_dim, args.pcl_r, args.moco_m, args.temperature, args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_margin = nn.TripletMarginLoss(margin=args.margin, p=2).cuda(args.gpu)

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=args.eps  # args.adam_epsilon  - default is 1e-8.
                      )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    sentence_train = json.load(open(args.data + 'tacred_train_sentence.json', 'r'))
    sentence_train_label = json.load(open(args.data + 'tacred_train_label_id.json', 'r'))
    sentence_train_must_links = json.load(open(args.data + 'tacred_train_pairs_3_c2ori_ds050.json', 'r'))
    sentence_train_entity_aug = json.load(open(args.data + 'tacred_train_entity_aug_sentence.json', 'r'))
    #sentence_train_entity_aug = json.load(open(args.data + 'tacred_train_sentence.json', 'r'))
    #print(len(sentence_train))
    #print(len(sentence_train_must_links))
    #print(len(sentence_train_entity_aug))
    #print()

    train_dataset, eval_dataset = pre_processing(sentence_train, sentence_train_label, sentence_train_must_links, sentence_train_entity_aug, args)
    # print(sentence_train)
    # print(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, sampler=train_sampler, drop_last=True)

    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers)

    gold = []
    for eval_data in eval_dataset:
        gold.append(eval_data[2].item())
    measure_file = open(os.path.join(args.exp_dir,'measurements_margin.txt'),'w')
    measure_file.write('Args: {}\n'.format(args))

    for epoch in range(args.start_epoch, args.epochs):

        cluster_result = None
        # compute clusters before every training
        if epoch >= args.warmup_epoch:
            measure_file.write('Epoch :'+str(epoch)+'\n')
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)

            # placeholder for clustering result
            cluster_result = {'relation2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in args.num_cluster:
                cluster_result['relation2cluster'].append(torch.zeros(len(eval_dataset), dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), args.low_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

            if args.gpu == 0:
                features[
                    torch.norm(features, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
                features = features.numpy()
                cluster_start = time.time()
                run_which = 'Kmeans'
                if args.repeat_index % 2 == 1:
                    cluster_result = run_kmeans(features, args)
                else:
                    cluster_result = calculate_hi_kmeans(features, args)
                    tmp = copy.deepcopy(cluster_result['relation2cluster'])
                    json_str = json.dumps([item.cpu().numpy().tolist() for item in tmp], indent=4)
                    with open("{}epoch_cluster_result.json".format(epoch), 'w') as json_file:
                        json_file.write(json_str)
                    run_which='AP'
                cluster_end = time.time()
                print('{} Running time: {} Seconds'.format(run_which, cluster_end - cluster_start))
                measure_file.write('{} epoch:{} Running time: {} Seconds\n'.format(run_which,epoch,cluster_end - cluster_start))
                # saving cluster measurements
                calculate_measurements(cluster_result, args, gold, measure_file, features)

            dist.barrier()
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:
                    # print(data_tensor)
                    dist.broadcast(data_tensor, 0, async_op=False)

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train(train_loader, model, criterion, criterion_margin, optimizer, epoch, args, cluster_result)

        max_save_times = 5  # max saving times, for saving storage space
        save_freq = int(args.epochs/max_save_times)
        save_freq = save_freq if save_freq>0 else int(args.epochs/max_save_times)
        if (epoch + 1) % save_freq == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                              and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_revised_{:04d}.pth.tar'.format(args.exp_dir, epoch+1))

    measure_file.close()


def train(train_loader, model, criterion, criterion_margin, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    #acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # batch.requires_grad=True
        if args.gpu is not None:
            for b_i in range(len(batch)):
                batch[b_i] = batch[b_i].cuda(args.gpu, non_blocking=True)


        # compute output
        output, target, ml_triplet_lst, output_proto, target_proto = model(sen_q=batch[:6], sen_k=batch[:5]+[batch[6]], cluster_result=cluster_result, index=batch[7], print_index=i, sen_aug=batch[8:13], must_links_l=batch[13:18], must_links_r=batch[18:22]+[batch[-1]], margin_info=True)
        # output.requires_grad=True

        # same-sentence pairs loss
        loss = 0
        if (output is not None) and len(output) > 0:
            loss_info = 0
            for triplet in output:
                loss_info += criterion_margin(triplet[0], triplet[1], triplet[2])
            loss_info /= len(output)
            loss += loss_info
        # loss.backward()

        # cross-sentence averaged triplet loss
        if (ml_triplet_lst is not None) and len(ml_triplet_lst) > 0:
            loss_must_link = 0
            for triplet in ml_triplet_lst:
                loss_must_link += criterion_margin(triplet[0], triplet[1], triplet[2])
            loss_must_link /= len(ml_triplet_lst)
            loss += loss_must_link

        # Hierarchical ExemNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)
                # loss_proto.backward()
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp[0], batch[0].size(0))

            # average loss across all sets of prototypes
            loss_proto /= len(output_proto)
            loss += loss_proto

        losses.update(loss.item(), batch[0].size(0))
        epoch_loss += loss.item()
        #acc = accuracy(output, target)[0]
        #acc_inst.update(acc[0], batch[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return epoch_loss/(i+1)


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, sentence in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            # sentence = sentence.cuda(non_blocking=True)
            for j in range(len(sentence)):
                sentence[j] = sentence[j].cuda(non_blocking=True)
            # pdb.set_trace()
            feat = model(sen_q=sentence,is_eval=True,print_index=i)
            features[sentence[-1]] = feat    # .view(args.low_dim*args.batch_size)
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()

# --------------------------  functions  ----------------------------

# --------------------------  for evaluation  ----------------------------
def calculate_measurements(cluster_result,args,gold,measure_file,features):
    """
    Calculate and save cluster measurements
    """
    def print_output(content):
        measure_file.write(content)
        measure_file.write('\n')
        print(content)
        pass
    count = 0
    for index, cluster_instance in enumerate(cluster_result['relation2cluster']):
        print_output('Cluster num: {}'.format(len(cluster_result['centroids'][index])))
        count += 1
        pred = cluster_instance.cpu().numpy()
        
        p, r, f1 = scorer.bcubed_score(gold, pred)
        print_output('B cube: p={:.5f} r={:.5f} f1={:.5f}'.format(p, r, f1))
        
        ari = scorer.adjusted_rand_score(gold, pred)
        homo, comp, v_m = scorer.v_measure(gold, pred)
        print_output('V-measure: hom.={:.5f} com.={:.5f} vm.={:.5f}'.format(homo, comp, v_m))
        print_output('ARI={:.5f}'.format(ari))
        
        ch_index = metrics.calinski_harabasz_score(features, pred)
        print_output('Calinski-Harabasz index={:.5f}'.format(ch_index))

        db_score = metrics.davies_bouldin_score(features, pred)
        print_output('Davies-Bouldin Index={:.5f}'.format(db_score))

        print_output('')

    measure_file.write('\n')
    pass

# --------------------------  preparing data  ----------------------------
def random_select_close_words(target_index, start, end, add_num):
    """
    Randomly selecting words between two entities which are closed to them
    """
    selected_index = [target_index]
    close_degree = 4
    while close_degree > 0 and len(selected_index) <= add_num:
        selected_range = [i for i in range(int(target_index - (target_index - start) / close_degree),
                          int(target_index + (end - target_index) / close_degree)) if i not in selected_index]
        selected_index = selected_index + selected_range[:add_num]
        close_degree -= 1
    if len(selected_index) <= add_num:
        selected_index += [target_index]*(add_num-len(selected_index))
    return selected_index[1:target_index+1]
    pass


def allnStopPos(encoded_dict_input_ids, stp_words_ids):
    result = []
    iter_id = 0
    for ed_id in encoded_dict_input_ids.data[0]:
        ed_id_item = ed_id.item()
        #if iter_id == 0:
        #    print(ed_id_item)
        if ed_id_item not in stp_words_ids:
            result.append(iter_id)
        iter_id += 1
    return result


# added constraints: should be non-stop words
def word_level_augmentation(pos1, pos2, pos1e, pos2e, validate_length, args, nstp_pos):
    """
    Data augmentation at word level
    """
    if pos1 > pos2:
        tmp = pos2
        pos2 = pos1
        pos1 = tmp
        tmp = pos2e
        pos2e = pos1e
        pos1e = tmp
    add_num = args.add_word_num
    middle_range = []
    for i in range(pos1e + 1, pos2):
        if i in nstp_pos:
            middle_range.append(i)
    random.shuffle(middle_range)
    selected_index = middle_range[:add_num]
    # if there are not enough words in the middle
    if len(selected_index) < add_num:
        add_num = add_num-len(selected_index)
        one_num = int(add_num/2)
        selected_index += random_select_close_words(pos1, 0, pos1, one_num)
        selected_index += random_select_close_words(pos2e, pos2e, validate_length, add_num - one_num)
    return selected_index
    pass


def get_tokenizer(args):
    """ Tokenize all of the sentences and map the tokens to their word IDs."""
    tokenizer = BertTokenizer.from_pretrained('../AugURE-master/bert-base-uncased', do_lower_case=True)
    special_tokens = []
    if not args.use_relation_span:
        special_tokens.append('<e1>')
        special_tokens.append('</e1>')
        special_tokens.append('<e2>')
        special_tokens.append('</e2>')
    else:
        #ent_type = ['LOCATION','MISC','ORGANIZATION','PERSON']   # NYT
        ent_type = ['PERSON', 'ORGANIZATION', 'NUMBER', 'DATE', 'NATIONALITY', 'LOCATION', 'TITLE', 'CITY', 'MISC', 'COUNTRY', 'CRIMINAL_CHARGE', 'RELIGION', 'DURATION', 'URL', 'STATE_OR_PROVINCE', 'IDEOLOGY', 'CAUSE_OF_DEATH'] #Tacred

        for r in ent_type:
            special_tokens.append('<e1:'+ r +'>')
            special_tokens.append('<e2:' + r + '>')
            special_tokens.append('</e1:' + r + '>')
            special_tokens.append('</e2:' + r + '>')
    special_tokens_dict ={'additional_special_tokens': special_tokens }    # add special token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def get_token_word_id(sen,tokenizer,args):
    """ Get special token word id """
    if not args.use_relation_span:
        # return 2487,2475
        e1 = '<e1>'
        e2 = '<e2>'
    else:
        e1 = re.search('(<e1:.*?>)', sen).group(1)
        e2 = re.search('(<e2:.*?>)', sen).group(1)
        e1e = re.search('(</e1:.*?>)', sen).group(1)
        e2e = re.search('(</e2:.*?>)', sen).group(1)
    e1_tks_id = tokenizer.convert_tokens_to_ids(e1)
    e2_tks_id = tokenizer.convert_tokens_to_ids(e2)
    e1e_id = tokenizer.convert_tokens_to_ids(e1e)
    e2e_id = tokenizer.convert_tokens_to_ids(e2e)
    # print('id:',e1_tks_id,'   ',e2_tks_id)
    return e1_tks_id, e2_tks_id, e1e_id, e2e_id
    pass


def label_sampling(sentence_train, sentence_train_label):
    """ Label sampling when doing case study or experiment """
    sampling_rate = 0.045
    random.seed(10)
    total_dict = {}
    for i,v in enumerate(sentence_train_label):
        if v not in total_dict:
            total_dict[v] = [i]
        else:
            total_dict[v].append(i)
    sampling_label = []
    for key in total_dict:
        kd = total_dict[key]
        s = random.sample(kd, int(sampling_rate*len(kd)))  # randomly select
        sampling_label+=s
    sampling_label.sort()
    nst = []
    nstl = []
    for i in sampling_label:
        nst.append(sentence_train[i])
        nstl.append(sentence_train_label[i])
    return nst,nstl
    pass


def pre_processing(sentence_train, sentence_train_label, sentence_train_must_links, sentence_train_entity_aug, args):
    """Main function for pre-processing data """
    # sentence_train,sentence_train_label = label_sampling(sentence_train,sentence_train_label)
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []
    aug_pos_arr1 = []
    aug_pos_arr2 = []
    aug_pos_arr3 = []
    index_arr = []

    aug_input_ids = []
    aug_attention_masks = []
    aug_e1_pos = []
    aug_e2_pos = []
    aug_aug_pos_arr1 = []

    l_ml_input_ids_lst = []
    l_ml_attention_masks_lst = []
    l_ml_e1_pos_lst = []
    l_ml_e2_pos_lst = []
    l_ml_aug_pos_arr_lst = []
    ml_input_ids_lst = []
    ml_attention_masks_lst = []
    ml_e1_pos_lst = []
    ml_e2_pos_lst = []
    ml_aug_pos_arr_lst = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = get_tokenizer(args)
    counter = 0
    # pre-processing sentenses to BERT pattern
    train_sen_file = open('train_sen_file.csv','w')
    train_sen_file.write('{}\t{}\t{}\n'.format('index','sentence','label'))
    train_sen_arr = []

    stop_words = ['the', 'to', 'and', 'a', 'an', 'in', 'it', 'is', 'are', 'of', 'I', 'that',
                  'had', 'on', 'for', 'were', 'was', 'from', 'by', 'with', 'have', 'has', 'be']

    stop_words_ids = []
    for stopw in stop_words:
        stopid = tokenizer.convert_tokens_to_ids(stopw)
        stop_words_ids.append(stopid)

    for i in range(len(sentence_train)):
        #print(i)
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            truncation=True,        # explicitely truncate examples to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        aug_encoded_dict = tokenizer.encode_plus(
            sentence_train_entity_aug[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            truncation=True,  # explicitely truncate examples to max length
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        l_ml_input_ids = []
        l_ml_attention_masks = []
        l_ml_e1_pos = []
        l_ml_e2_pos = []
        l_ml_aug_pos_arr = []
        ml_input_ids = []
        ml_attention_masks = []
        ml_e1_pos = []
        ml_e2_pos = []
        ml_aug_pos_arr = []

        try:
            # Find e1(id:2487) and e2(id:2475) position
            e1_tks_id, e2_tks_id, e1e_id, e2e_id = get_token_word_id(sentence_train[i], tokenizer, args)
            pos1 = (encoded_dict['input_ids'] == e1_tks_id).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == e2_tks_id).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            pos1e = (encoded_dict['input_ids'] == e1e_id).nonzero()[0][1].item()
            pos2e = (encoded_dict['input_ids'] == e2e_id).nonzero()[0][1].item()

            # return all positions of non-stop words
            nstop_pos = allnStopPos(encoded_dict['input_ids'], stop_words_ids)

            # data augmentation
            aug_pos = word_level_augmentation(pos1, pos2, pos1e, pos2e, encoded_dict['input_ids'].nonzero().shape[0], args, nstop_pos)
            aug_pos_arr1.append(aug_pos)
            aug_pos = word_level_augmentation(pos1, pos2, pos1e, pos2e, encoded_dict['input_ids'].nonzero().shape[0], args, nstop_pos)
            aug_pos_arr2.append(aug_pos)
            aug_pos = word_level_augmentation(pos1, pos2, pos1e, pos2e, encoded_dict['input_ids'].nonzero().shape[0], args, nstop_pos)
            aug_pos_arr3.append(aug_pos)

            # for entity aug sentence
            aug_e1_tks_id, aug_e2_tks_id, aug_e1e_id, aug_e2e_id = get_token_word_id(sentence_train_entity_aug[i], tokenizer, args)
            aug_pos1 = (aug_encoded_dict['input_ids'] == aug_e1_tks_id).nonzero()[0][1].item()
            aug_pos2 = (aug_encoded_dict['input_ids'] == aug_e2_tks_id).nonzero()[0][1].item()
            aug_pos1e = (aug_encoded_dict['input_ids'] == aug_e1e_id).nonzero()[0][1].item()
            aug_pos2e = (aug_encoded_dict['input_ids'] == aug_e2e_id).nonzero()[0][1].item()

            aug_nstop_pos = allnStopPos(aug_encoded_dict['input_ids'], stop_words_ids)

            # data augmentation
            aug_aug_pos = word_level_augmentation(aug_pos1, aug_pos2, aug_pos1e, aug_pos2e, aug_encoded_dict['input_ids'].nonzero().shape[0], args, aug_nstop_pos)
            aug_aug_pos_arr1.append(aug_aug_pos)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
            index_arr.append(counter)
            train_sen_file.write('{}\t{}\t{}\n'.format(i, sentence_train[i], sentence_train_label[i]))
            train_sen_arr.append(sentence_train[i])
            counter += 1

            aug_input_ids.append(aug_encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            aug_attention_masks.append(aug_encoded_dict['attention_mask'])
            aug_e1_pos.append(aug_pos1)
            aug_e2_pos.append(aug_pos2)

            for j in range(len(sentence_train_must_links[i])):
                # left item of the pair
                l_ml_encoded_dict = tokenizer.encode_plus(
                    sentence_train_must_links[i][j][0],  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=args.max_length,  # Pad & truncate all sentences.
                    truncation=True,  # explicitely truncate examples to max length
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                # Find e1(id:2487) and e2(id:2475) position
                l_ml_e1_tks_id, l_ml_e2_tks_id, l_ml_e1e_id, l_ml_e2e_id = get_token_word_id(sentence_train_must_links[i][j][0], tokenizer, args)
                l_ml_pos1 = (l_ml_encoded_dict['input_ids'] == l_ml_e1_tks_id).nonzero()[0][1].item()
                l_ml_pos2 = (l_ml_encoded_dict['input_ids'] == l_ml_e2_tks_id).nonzero()[0][1].item()
                l_ml_e1_pos.append(l_ml_pos1)
                l_ml_e2_pos.append(l_ml_pos2)
                l_ml_pos1e = (l_ml_encoded_dict['input_ids'] == l_ml_e1e_id).nonzero()[0][1].item()
                l_ml_pos2e = (l_ml_encoded_dict['input_ids'] == l_ml_e2e_id).nonzero()[0][1].item()

                l_ml_nstop_pos = allnStopPos(l_ml_encoded_dict['input_ids'], stop_words_ids)

                # data augmentation
                l_ml_aug_pos = word_level_augmentation(l_ml_pos1, l_ml_pos2, l_ml_pos1e, l_ml_pos2e, l_ml_encoded_dict['input_ids'].nonzero().shape[0], args, l_ml_nstop_pos)
                l_ml_aug_pos_arr.append(l_ml_aug_pos)
                # Add the encoded sentence to the list.
                l_ml_input_ids.append(l_ml_encoded_dict['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                l_ml_attention_masks.append(l_ml_encoded_dict['attention_mask'])

                # right item of the pair
                ml_encoded_dict = tokenizer.encode_plus(
                    sentence_train_must_links[i][j][1],  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=args.max_length,  # Pad & truncate all sentences.
                    truncation=True,  # explicitely truncate examples to max length
                    pad_to_max_length=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                # Find e1(id:2487) and e2(id:2475) position
                ml_e1_tks_id, ml_e2_tks_id, ml_e1e_id, ml_e2e_id = get_token_word_id(sentence_train_must_links[i][j][1], tokenizer, args)
                ml_pos1 = (ml_encoded_dict['input_ids'] == ml_e1_tks_id).nonzero()[0][1].item()
                ml_pos2 = (ml_encoded_dict['input_ids'] == ml_e2_tks_id).nonzero()[0][1].item()
                ml_e1_pos.append(ml_pos1)
                ml_e2_pos.append(ml_pos2)
                ml_pos1e = (ml_encoded_dict['input_ids'] == ml_e1e_id).nonzero()[0][1].item()
                ml_pos2e = (ml_encoded_dict['input_ids'] == ml_e2e_id).nonzero()[0][1].item()

                ml_nstop_pos = allnStopPos(ml_encoded_dict['input_ids'], stop_words_ids)

                # data augmentation
                ml_aug_pos = word_level_augmentation(ml_pos1, ml_pos2, ml_pos1e, ml_pos2e, ml_encoded_dict['input_ids'].nonzero().shape[0], args, ml_nstop_pos)
                ml_aug_pos_arr.append(ml_aug_pos)
                # Add the encoded sentence to the list.
                ml_input_ids.append(ml_encoded_dict['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                ml_attention_masks.append(ml_encoded_dict['attention_mask'])

            l_ml_input_ids_lst.append(torch.cat(l_ml_input_ids, dim=0))
            l_ml_attention_masks_lst.append(torch.cat(l_ml_attention_masks, dim=0))
            l_ml_e1_pos_lst.append(torch.tensor(l_ml_e1_pos))
            l_ml_e2_pos_lst.append(torch.tensor(l_ml_e2_pos))
            l_ml_aug_pos_arr_lst.append(torch.tensor(l_ml_aug_pos_arr))

            ml_input_ids_lst.append(torch.cat(ml_input_ids, dim=0))
            ml_attention_masks_lst.append(torch.cat(ml_attention_masks, dim=0))
            ml_e1_pos_lst.append(torch.tensor(ml_e1_pos))
            ml_e2_pos_lst.append(torch.tensor(ml_e2_pos))
            ml_aug_pos_arr_lst.append(torch.tensor(ml_aug_pos_arr))

        except Exception as e:
            #print(sentence_train[i])
            print(e)
            pass
            #print(sent)


    # Convert the lists into tensors.
    json_str = json.dumps([train_sen_arr,labels], indent=4)
    with open("train_data_arr.json", 'w') as json_file:
        json_file.write(json_str)
    print(len(index_arr))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)
    aug_pos_arr1 = torch.tensor(aug_pos_arr1)
    aug_pos_arr2 = torch.tensor(aug_pos_arr2)
    aug_pos_arr3 = torch.tensor(aug_pos_arr3)
    index_arr = torch.tensor(index_arr)

    aug_input_ids = torch.cat(aug_input_ids, dim=0)
    aug_attention_masks = torch.cat(aug_attention_masks, dim=0)
    aug_e1_pos = torch.tensor(aug_e1_pos)
    aug_e2_pos = torch.tensor(aug_e2_pos)
    aug_aug_pos_arr1 = torch.tensor(aug_aug_pos_arr1)

    print()
    print(input_ids.size(0))
    print(attention_masks.size(0))
    print(e1_pos.size(0))
    print(input_ids.size())
    print(labels.size())


    l_ml_input_ids_lst = torch.stack(l_ml_input_ids_lst)
    l_ml_attention_masks_lst = torch.stack(l_ml_attention_masks_lst)
    l_ml_e1_pos_lst = torch.stack(l_ml_e1_pos_lst)
    l_ml_e2_pos_lst = torch.stack(l_ml_e2_pos_lst)
    l_ml_aug_pos_arr_lst = torch.stack(l_ml_aug_pos_arr_lst)

    ml_input_ids_lst = torch.stack(ml_input_ids_lst)
    ml_attention_masks_lst = torch.stack(ml_attention_masks_lst)
    ml_e1_pos_lst = torch.stack(ml_e1_pos_lst)
    ml_e2_pos_lst = torch.stack(ml_e2_pos_lst)
    ml_aug_pos_arr_lst = torch.stack(ml_aug_pos_arr_lst)

    print()
    print(ml_input_ids_lst.size(0))
    print(ml_attention_masks_lst.size(0))
    print(ml_e1_pos_lst.size(0))
    print(ml_input_ids_lst.size())
    print()

    eval_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, aug_pos_arr3, index_arr)

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, aug_pos_arr1, aug_pos_arr2, index_arr, \
                                  aug_input_ids, aug_attention_masks, aug_e1_pos, aug_e2_pos, aug_aug_pos_arr1, \
                                  l_ml_input_ids_lst,  l_ml_attention_masks_lst, l_ml_e1_pos_lst, l_ml_e2_pos_lst, l_ml_aug_pos_arr_lst, \
                                  ml_input_ids_lst,  ml_attention_masks_lst, ml_e1_pos_lst, ml_e2_pos_lst, ml_aug_pos_arr_lst)
    train_sen_file.close()
    return train_dataset, eval_dataset

# --------------------------  clustering methods  ----------------------------
def parse_clustering(x, k, relation2cluster, args):
    """Get centroids and density after clustering """
    cluster_group = [[] for ki in range(k)]
    Dcluster = [[] for c in range(k)]
    for li in range(len(relation2cluster)):
        cluster_group[relation2cluster[li]].append(x[li])

    centroids = []
    for gi, group in enumerate(cluster_group):
        new_cent = sum(group) / len(group)
        centroids.append(new_cent)
        group_dist = []
        for item in group:
            group_dist.append(np.linalg.norm(item - new_cent))
        # print(group_dist)
        Dcluster[gi] = group_dist

    print('centroids calculate')
    # concentration estimation (phi)
    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + args.alpha)
            density[i] = d

            # if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    if density.mean() != 0:
        density = args.temperature * density / density.mean()  # scale the mean to temperature
    # print('density calculate')
    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    relation2cluster = torch.LongTensor(relation2cluster).cuda()
    density = torch.Tensor(density).cuda()

    return centroids, relation2cluster, density


def kmeans_clusering(x,args):
    print('performing kmeans clustering')
    results = {'relation2cluster': [], 'centroids': [], 'density': []}
    for seed, num_cluster in enumerate(args.num_cluster):
        k = int(num_cluster)
        km_result = KMeans(n_clusters=k, random_state=0).fit(x)
        relation2cluster = km_result.labels_

        centroids, relation2cluster, density = parse_clustering(x, k, relation2cluster, args)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['relation2cluster'].append(relation2cluster)

    return results


def hi_kmeans(x, args):
    final_result = {'relation2cluster':[],'centroids':[],'density':[]}
    count = 0
    while count<3:
        cluster_result = run_kmeans(x, args)
        for index,cluster_instance in enumerate(cluster_result['relation2cluster']):
            pred = cluster_instance.cpu().numpy()
            ch_index = metrics.calinski_harabasz_score(x, pred)
            db_score = metrics.davies_bouldin_score(x, pred)
        count+=1
    return final_result


def merge_block(block1,group1,block2,group2):
    block1['centroids'] = torch.cat([block1['centroids'],block2['centroids']], 0) if len(block1['centroids'])>0 else block2['centroids']
    block1['density'] = torch.cat([block1['density'], block2['density']], 0) if len(block1['density'])>0 else block2['density']
    # block1['density'] += block2['density']

    n = len(group1)
    for i in range(len(block2['relation2cluster'])):
        block2['relation2cluster'][i]+=n

    block1['relation2cluster'] += block2['relation2cluster']
    group1+=group2

    return block1,group1
    pass

def rearrange(group,n):
    r2c = [0]*n
    for i, g in enumerate(group):
        for item in g:
            r2c[item] = i
    return torch.LongTensor(r2c).cuda()
    pass


def calculate_hi_kmeans(x, args):
    results = {'relation2cluster': [], 'centroids': [], 'density': []}

    from collections import deque
    q = deque()
    q.append(hierarchical_kmeans(x, args))
    max_level = 5
    count = 0

    while q and count < max_level:
        print('*'*20+'{} level'.format(count)+'*'*20)
        if results['centroids'] and len(results['centroids'][-1])>=10:
            break
        level_len = len(q)
        level_block, cluster_group = {'relation2cluster': [], 'centroids': [], 'density': []}, []
        for i in range(level_len):
            lb,cg = q.popleft()
            level_block, cluster_group = merge_block(level_block, cluster_group,lb,cg)
            pass
        level_block['relation2cluster'] = rearrange(cluster_group,len(x))
        for clu in cluster_group:
            if len(clu)>10:
                new_item = hierarchical_kmeans(x[clu], args)
                q.append(new_item)
            else:
                print('-'*20)
                # print(clu)

        results['centroids'].append(level_block['centroids'])
        results['density'].append(level_block['density'])
        results['relation2cluster'].append(level_block['relation2cluster'])
        count += 1

    return results
    pass


def hierarchical_kmeans(x, args):
    print('performing hierarchical_kmeans clustering')
    # results = {'relation2cluster': [], 'centroids': [], 'density': []}
    best_k = 0
    best_r2c = []
    db_index_max = float('inf')
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        x_attention = x
        d = x_attention.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        # clus.max_points_per_centroid = 1000
        # clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        # print(x_attention)
        clus.train(x_attention, index)

        D, I = index.search(x_attention, 1)  # for each sample, find cluster distance and assignments
        relation2cluster = [int(n[0]) for n in I]
        # ch_index = metrics.calinski_harabasz_score(x, relation2cluster)
        db_score = metrics.davies_bouldin_score(x, relation2cluster)
        if db_score<db_index_max:
            best_k = k
            best_r2c = relation2cluster
            db_index_max = db_score
            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im, i in enumerate(relation2cluster):
                Dcluster[i].append(D[im][0])


    # concentration estimation (phi)
    density = np.zeros(best_k)

    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + args.alpha)
            density[i] = d
        # centroids[i]
        group_dist_arr = []
        for j, dj in enumerate(dist):
            group_dist_arr.append(np.linalg.norm(dj - centroids[i]))

    # if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 10),
                           np.percentile(density, 90))  # clamp extreme values for stability
    if density.mean() != 0:
        density = args.temperature * density / density.mean()  # scale the mean to temperature

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    relation2cluster = torch.LongTensor(best_r2c).cuda()
    density = torch.Tensor(density).cuda()

    level_block = {}
    level_block['centroids']= centroids
    level_block['density'] = density
    level_block['relation2cluster']= relation2cluster

    cluster_group = [[] for _ in range(best_k)]
    for i, r in enumerate(best_r2c):
        cluster_group[r].append(i)

    return level_block,cluster_group
    pass


def x_centroids_attention(centroids,relation2cluster,x):
    lamb=1
    similar_matrix = torch.mm(lamb*centroids,centroids.T)

    attended = F.softmax(similar_matrix,dim=1)
    x_hat = torch.einsum('ij,jk->ik', [attended, centroids])
    x_hat=x_hat.cpu().numpy()
    m=0.9
    for i,r2c in enumerate(relation2cluster):
        x[i]=0.9*x[i]+x_hat[r2c]*(1-m)
    return x
    try:
        centroids = centroids.cpu().numpy()
        relation2cluster = relation2cluster.cpu().numpy()
    except:
        pass

    y = []
    for i, r2c in enumerate(relation2cluster):
        y.append(np.dot(x[i], centroids[r2c])*x[i])
    return np.array(y)
    pass

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'relation2cluster':[], 'centroids':[], 'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        x_attention = x
        if seed > 0 :#and args.attention
            x_attention = x_centroids_attention(results['centroids'][seed-1],results['relation2cluster'][seed-1],x)
            pass
        d = x_attention.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        # print(x_attention)
        clus.train(x_attention, index)

        D, I = index.search(x_attention, 1)  # for each sample, find cluster distance and assignments
        relation2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]
        Dcluster_index = [[] for c in range(k)]
        for im,i in enumerate(relation2cluster):
            Dcluster[i].append(D[im][0])
            Dcluster_index[i].append(im)
        
        # concentration estimation (phi)        
        density = np.zeros(k)

        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+args.alpha)
                density[i] = d
            # centroids[i]
            group_dist_arr = []
            for j, dj in enumerate(dist):
                group_dist_arr.append(np.linalg.norm(dj - centroids[i]))

        # train_sen_file.close()
        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        if density.mean()!=0:
            density = args.temperature*density/density.mean()  #scale the mean to temperature
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        relation2cluster = torch.LongTensor(relation2cluster).cuda()
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['relation2cluster'].append(relation2cluster)
        
    return results


# --------------------------  utils  ----------------------------
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    print('lr: ',lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
