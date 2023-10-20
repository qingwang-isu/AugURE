import json
import numpy as np
import random

def keep(s):
    pre_ent_type = ['LOCATION', 'MISC', 'ORGANIZATION', 'PERSON']
    lst = s.split(" ")
    for item in lst:
        if item.startswith("<e1"):
            tmp_type = item.lstrip("<e1:").rstrip(">")
            if tmp_type not in pre_ent_type:
                return False
        elif item.startswith("<e2"):
            tmp_type = item.lstrip("<e2:").rstrip(">")
            if tmp_type not in pre_ent_type:
                return False
    return True


def enumerateAllPairs(lst, template):
    result = []
    for a in range(len(lst)-1):
        for b in range(a+1, len(lst)):
            result.append([lst[a], lst[b], template])
    return result


def mutualAllPairs(lst1, lst2, template1, template2):
    result = []
    for a in range(len(lst1)):
        for b in range(len(lst2)):
            result.append([lst1[a], lst2[b], template1, template2])
    return result


inF = open("output/tacred_train_chatGPT_typed_template_instance_dict_3.json", "r")
template_instance_dict = json.load(inF)
inF.close()

inF_all = open("output/tacred_train_chatGPT_typed_template_instance_dict.json", "r")
template_instance_dict_all = json.load(inF_all)
inF_all.close()

inF2 = open('output/tacred_train_chatGPT_typed_entail_edge_weight_matrix_3.txt', 'r')
edge_weight_matrix = np.loadtxt(inF2).reshape((len(template_instance_dict), len(template_instance_dict)))


inF_chatGPT = open('../data_sample_for_exemple/tacred_train_chatGPT_sentence.json', 'r')
chatGPT_lst = json.load(inF_chatGPT)
chatGPT_s2id = {}
for i in range(0, len(chatGPT_lst)):
    chatGPT_s2id[chatGPT_lst[i]] = i


# total 6626 sentences
# 5194 non-duplicated pairs
ori_pair_F = open('../data_sample_for_exemple/tacred_train_pairs_3.json', 'r')
ori_pairs_p = json.load(ori_pair_F)
ori_pair_F.close()


ori_pairs = []
cnt = 0
for p_pair in ori_pairs_p:
    if cnt == 5194:
        break
    ori_pairs.append(p_pair[0])
    cnt += 1
print(ori_pairs[0])
print(len(ori_pairs))
pairs = []

# generate in-template must-link pairs
for key in template_instance_dict_all:
    instance_lst = template_instance_dict_all[key]
    tmp_pairs = enumerateAllPairs(instance_lst, key)
    pairs.extend(tmp_pairs)
print("# same-template positive pairs:")
prev = len(pairs)
print(prev)


id2template_dict = {}
id_cnt = 0
for key in template_instance_dict:
    id2template_dict[id_cnt] = key
    num_instances = len(template_instance_dict[key])
    id_cnt += 1

# generate mutual-entail template pairs
prob_threshold = 0.95
for i in range(len(id2template_dict)-1):
    for j in range(i+1, len(id2template_dict)):
        if edge_weight_matrix[i][j] >= prob_threshold and edge_weight_matrix[j][i] >= prob_threshold:
            t_i = id2template_dict[i]
            t_j = id2template_dict[j]
            tmp_pairs = mutualAllPairs(template_instance_dict[t_i], template_instance_dict[t_j], t_i, t_j)
            pairs.extend(tmp_pairs)
print("# cross-templates positive pairs:")
#print(len(pairs))
print(len(pairs)-prev)


ori_train_F = open("../data_sample_for_exemple/tacred_train_sentence.json", "r")
ori_lst = json.load(ori_train_F)
num_ori = len(ori_lst)
print(num_ori)

print("# total positive pairs:")
print(len(pairs))

random.shuffle(pairs)
#down_sampling
new_pairs = []
target_size = 18446
print("size after sampling")
print(target_size)
cnt = 0
iter = 0
while (cnt < target_size):
    if cnt % 1000 == 0:
        print(cnt)
    p = pairs[iter]
    iter += 1
    id0 = chatGPT_s2id[p[0]]
    id1 = chatGPT_s2id[p[1]]
    if "<e1" not in ori_lst[id0] or "<e2" not in ori_lst[id0]:
        continue
    if "<e1" not in ori_lst[id1] or "<e2" not in ori_lst[id1]:
        continue
    tmp_pair = [ori_lst[id0], ori_lst[id1]]
    if tmp_pair not in ori_pairs:
        new_pairs.append(tmp_pair)
        cnt += 1
print("# total added positive pairs (compared with original pairs):")
num_pairs = len(new_pairs)
print(num_pairs)

k = int((len(ori_pairs)+num_pairs)/num_ori)
print(k)
rem = (len(ori_pairs)+num_pairs)-k*num_ori

print(new_pairs[0])
random.shuffle(new_pairs)
print(new_pairs[0])

pairs_01 = []
for p in ori_pairs:
    pairs_01.append([p[0], p[1]])
for r in range(0, num_pairs):
    pairs_01.append([new_pairs[r][0], new_pairs[r][1]])

print(len(pairs_01))
print(pairs_01[0])
random.shuffle(pairs_01)
print(pairs_01[0])


formatted_out = []
for sid in range(num_ori):
    tmp_lst = []
    tmp_lst.extend(pairs_01[(sid*k):(sid*k+k)])
    formatted_out.append(tmp_lst)


add_id = 0
while rem > 0:
    tmp_pair = pairs_01[len(pairs_01)-rem]
    rem -= 1
    formatted_out[add_id].append(tmp_pair)
    add_id += 1


print(len(formatted_out))
print(len(formatted_out[0]))
print(len(formatted_out[6000]))
json.dump(formatted_out, open('../data_sample_for_exemple/tacred_train_pairs_3_c2ori_equal.json', 'w'))
