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


inF = open("output/tacred_train_template_instance_dict_3.json", "r")
template_instance_dict = json.load(inF)
inF.close()

inF_all = open("output/tacred_train_template_instance_dict.json", "r")
template_instance_dict_all = json.load(inF_all)
inF_all.close()

inF2 = open('output/tacred_train_entail_edge_weight_matrix_3.txt', 'r')
edge_weight_matrix = np.loadtxt(inF2).reshape((len(template_instance_dict), len(template_instance_dict)))


pairs = []
# generate in-template must-link pairs
for key in template_instance_dict_all:
    instance_lst = template_instance_dict_all[key]
    tmp_pairs = enumerateAllPairs(instance_lst, key)
    pairs.extend(tmp_pairs)
print("# in-template must-links:")
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
print("# cross-templates must-links:")
print(len(pairs)-prev)


ori_train_F = open("../data_sample_for_exemple/tacred_train_sentence.json", "r")
ori_dict = json.load(ori_train_F)
num_ori = len(ori_dict)
print(num_ori)

num_pairs = len(pairs)
print(num_pairs)

k = int(num_pairs/num_ori)
print(k)

print(pairs[0])
random.shuffle(pairs)
print(pairs[0])

pairs_01 = []
for p in pairs:
    pairs_01.append([p[0], p[1]])

random.shuffle(pairs)
pairs_02 = []
for p in pairs:
    pairs_02.append([p[0], p[1]])


formatted_out = []
if k > 0:
    for sid in range(num_ori):
        tmp_lst = []
        tmp_lst.extend(pairs_01[(sid*k):(sid*k+k)])
        formatted_out.append(tmp_lst)
else:
    num_pairs = len(pairs_01)
    for sid in range(num_ori):
        tmp_lst = []
        if sid < num_pairs:
            tmp_lst.append(pairs_01[sid])
        elif sid < 2*num_pairs:
            tmp_lst.append(pairs_02[sid-num_pairs])
        else:
            rd_key = random.randint(0, num_pairs-1)
            tmp_lst.append(pairs_01[rd_key])
        formatted_out.append(tmp_lst)

print(len(formatted_out))
json.dump(formatted_out, open('output/tacred_train_pairs_3.json', 'w'))
