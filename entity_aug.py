import json
import random

def getTypedEntities(sen):
    tokens = sen.split(" ")
    e1 = ""
    e2 = ""
    type1 = ""
    type2 = ""
    in1 = False
    in2 = False
    for i in range(len(tokens)):
        tk = tokens[i]
        if "<e1:" in tk:
            type1 += tk.lstrip("<e1:").rstrip(">")
            in1 = True
        elif "</e1:" in tk:
            in1 = False
        elif "<e2:" in tk:
            type2 += tk.lstrip("<e2:").rstrip(">")
            in2 = True
        elif "</e2:" in tk:
            in2 = False

        if ("<e1:" not in tk) and (in1 == True):
            e1 += tk
            e1 += " "
        if ("<e2:" not in tk) and (in2 == True):
            e2 += tk
            e2 += " "
    return type1, e1.rstrip(" "), type2, e2.rstrip(" ")



def getReplacedS(sen, type, old_entity, dict):
    new_id = random.randrange(len(dict[type]))
    new_entity = dict[type][new_id]
    new_sen = sen.replace(old_entity, new_entity)
    return new_sen

inF = open("../data_sample_for_exemple/tacred_train_sentence.json", "r")
lst = json.load(inF)
inF.close()
print(len(lst))

# dictionary of entity type & all entities
entity_dict = {}
for s1 in lst:
    t1, ent1, t2, ent2 = getTypedEntities(s1)
    if t1 not in entity_dict:
        entity_dict[t1] = []
        entity_dict[t1].append(ent1)
    else:
        entity_dict[t1].append(ent1)
    if t2 not in entity_dict:
        entity_dict[t2] = []
        entity_dict[t2].append(ent2)
    else:
        entity_dict[t2].append(ent2)

#print(entity_dict)

total_cnt = 0
result_lst = []
for s in lst:
    t1, ent1, t2, ent2 = getTypedEntities(s)
    rd_num = random.randrange(4)
    # replace e1
    if rd_num % 2 == 0:
        rps = getReplacedS(s, t1, ent1, entity_dict)
        result_lst.append(rps)
        total_cnt += 1
    # replace e2
    else:
        rps = getReplacedS(s, t2, ent2, entity_dict)
        result_lst.append(rps)
        total_cnt += 1


with open("../data_sample_for_exemple/tacred_train_entity_aug_sentence.json", "w") as f:
    json.dump(result_lst, f)

print(total_cnt)