import time
import json
from allennlp_models.pretrained import load_predictor
from pyvis.network import Network
import networkx as nx
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer


def extractSentences(lst):
    sentences = []
    for tp in lst:
        sentences.append(tp['sentence'])
    return sentences


def replaceTK(input_s):
    tokens = input_s.split(" ")
    rp_tokens = []

    ise1 = False
    ise2 = False
    for tk in tokens:
        if ise1:
            if tk.startswith("</e1:"):
                ise1 = False
                continue
        elif ise2:
            if tk.startswith("</e2:"):
                ise2 = False
                continue
        else:
            if tk.startswith("<e1:"):
                ise1 = True
                rp_tokens.append(SUB_TK)
                continue
            elif tk.startswith("<e2:"):
                ise2 = True
                rp_tokens.append(OBJ_TK)
                continue
            else:
                rp_tokens.append(tk)
            
    return rp_tokens
 
    
# get head, tail, and clean tokens from a sentence like:
# Although the first staging dates from 1892 , the '' Nutcracker '' craze began with <e1:PERSON> George Balanchine </e1:PERSON> 's 1954 production for the <e2:LOCATION> New York City Ballet </e2:LOCATION> .
def getFormattedItems(input_s):
    tokens = input_s.split(" ")
    he = ""
    te = ""
    clean_tokens = []
    ise1 = False
    ise2 = False
    for tk in tokens:
        if ise1:
            if tk.startswith("</e1:"):
                ise1 = False
                continue
            else:
                if he != "":
                    he += " "
                he += tk
                clean_tokens.append(tk)
        elif ise2:
            if tk.startswith("</e2:"):
                ise2 = False
                continue
            else:
                if te != "":
                    te += " "
                te += tk
                clean_tokens.append(tk)
        else:
            if tk.startswith("<e1:"):
                ise1 = True
                continue
            elif tk.startswith("<e2:"):
                ise2 = True
                continue
            else:
                clean_tokens.append(tk)
    return he.lower(), te.lower(), clean_tokens   
    

def getTypes(typed_template):
    tk_lst = typed_template.split(" ")
    tail_id = len(tk_lst)-1
    t1 = tk_lst[0]
    t2 = tk_lst[tail_id]
    template = ""
    for cur_id in range(1, tail_id):
        template += tk_lst[cur_id]
        template += " "
    return t1, t2, template.rstrip(" ")



SUB_TK = "[h]"
OBJ_TK = "[t]"

inF = open("output/train_chatGPT_typed_template_instance_dict.json", "r")
template_instance_dict = json.load(inF)
inF.close()

# filtering
min_freq = 5
new_template_instance_dict = {}
for key in template_instance_dict:
    l = len(template_instance_dict[key])
    if l >= min_freq:
        new_template_instance_dict[key] = template_instance_dict[key]
json.dump(new_template_instance_dict, open('output/train_chatGPT_typed_template_instance_dict_5.json', 'w'))

        
id2template_dict = {}
template_s_dict = {}
id2label_dict = {}

net = Network(directed=True)
# G is a networkx graph and is used to save to gml file 
G = nx.DiGraph()

# add all template nodes
id_cnt = 0
for key in new_template_instance_dict:
    id2template_dict[id_cnt] = key
    #template_s_dict[key] = extractSentences(template_instance_dict[key]) 
    num_instances = len(new_template_instance_dict[key])
    net.add_node(id_cnt, label=key+"["+str(num_instances)+"]")
    G.add_node(key+"["+str(num_instances)+"]")
    id2label_dict[id_cnt] = key+"["+str(num_instances)+"]"
    id_cnt += 1

    
p = load_predictor("pair-classification-roberta-snli")
detokenizer = TreebankWordDetokenizer()


majority_threshold = 0.95
edge_weight_matrix = np.zeros((len(new_template_instance_dict), len(new_template_instance_dict)))

overall_stime = time.time()
printone = True
# edge i to j (hypo to prem, reverse entailment direction)
print(len(id2template_dict))
print()

for i in range(0, len(id2template_dict)):
    t_hypo_template = id2template_dict[i]
    #print(t_hypo_template)
    hypo_type1, hypo_type2, hypo_template = getTypes(t_hypo_template)
    print("processing node "+str(i)+"...")
    start = time.time()

    for j in range(0, len(id2template_dict)):
        if i != j:
            t_prem_template = id2template_dict[j]
            #sentences_j = template_s_dict[t_prem_template]
            instances = new_template_instance_dict[t_prem_template]
            prem_type1, prem_type2, prem_template = getTypes(t_prem_template)
            
            # if the entity types do not match, skip
            if (hypo_type1 != prem_type1) or (hypo_type2 != prem_type2):
                continue
            
            num_total = len(instances)
            num_entail = 0
            num_not = 0
            max_not = int(num_total*0.05)
            sum_probs = 0

            for k in range(0, num_total):
                triple = instances[k]
                sub, obj, tokens = getFormattedItems(instances[k])
                # replace the head & tail by [h] & [t] 
                rp_tokens = replaceTK(instances[k])
                s = detokenizer.detokenize(rp_tokens).lower()
                prem = s
                #hypo = sub + " " + hypo_template + " " + obj
                hypo = SUB_TK + " " + hypo_template + " " + OBJ_TK
                if printone == True:
                    print(prem)
                    print(hypo)
                    printone = False
                result = p.predict(premise=prem, hypothesis=hypo)
                if result['label'] == 'entailment':
                    num_entail += 1
                else:
                    num_not += 1
                    if num_not > max_not:
                        break
                sum_probs += result['probs'][0]
                
            if num_entail/num_total >= majority_threshold:
                ave_probs = sum_probs/num_total
                net.add_edge(i, j, label=str(format(ave_probs, '.3f')))
                G.add_edge(id2label_dict[i], id2label_dict[j], weight=format(ave_probs, '.3f'))
                edge_weight_matrix[i][j] = format(ave_probs, '.3f')
            
    end = time.time()
    print("used "+str(end-start)+"s")

overall_etime = time.time()

net.show_buttons(filter_=True)

net.show('train_chatGPT_5_typed.html')
nx.write_gml(G, 'train_chatGPT_5_typed.gml')

with open('output/train_chatGPT_typed_entail_edge_weight_matrix_5.txt', 'w') as outF:
    np.savetxt(outF, edge_weight_matrix)

print()
print("Total runtime: "+str((overall_etime-overall_stime)/60)+" minutes" )
