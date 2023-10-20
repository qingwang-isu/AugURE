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


def replaceTK(tokens, head, tail):
    rp_tokens = []
    e1_begin = int(head['e1_begin'])
    e1_end = int(head['e1_end'])
    e2_begin = int(tail['e2_begin'])
    e2_end = int(tail['e2_end'])
    
    for t in range(len(tokens)):
        if t == e1_begin:
            rp_tokens.append(SUB_TK)
        elif t > e1_begin and t <= e1_end:
            continue
        elif t == e2_begin:
            rp_tokens.append(OBJ_TK)
        elif t > e2_begin and t <= e2_end:
            continue
        else:
            rp_tokens.append(tokens[t])
            
    return rp_tokens
    
    

SUB_TK = "[h]"
OBJ_TK = "[t]"

inF = open("output/h_test_test_template_instance_dict_all2test.json", "r")
template_instance_dict = json.load(inF)
inF.close()

id2template_dict = {}
template_s_dict = {}
id2label_dict = {}

net = Network(directed=True)
# G is a networkx graph and is used to save to gml file 
G = nx.DiGraph()

# add all template nodes
id_cnt = 0
for key in template_instance_dict:
    id2template_dict[id_cnt] = key
    #template_s_dict[key] = extractSentences(template_instance_dict[key]) 
    num_instances = len(template_instance_dict[key])
    net.add_node(id_cnt, label=key+"["+str(num_instances)+"]")
    G.add_node(key+"["+str(num_instances)+"]")
    id2label_dict[id_cnt] = key+"["+str(num_instances)+"]"
    id_cnt += 1

    
p = load_predictor("pair-classification-roberta-snli")
detokenizer = TreebankWordDetokenizer()

majority_threshold = 0.95
edge_weight_matrix = np.zeros((len(template_instance_dict), len(template_instance_dict)))

overall_stime = time.time()
printone = True
# edge i to j (hypo to prem, reverse entailment direction)
for i in range(0, len(id2template_dict)):
    hypo_template = id2template_dict[i]
    print("processing node "+str(i)+"...")
    start = time.time()
    #sentences_i = template_s_dict[hypo_template]

    for j in range(0, len(id2template_dict)):
        if i != j:
            prem_template = id2template_dict[j]
            #sentences_j = template_s_dict[prem_template]
            instances = template_instance_dict[prem_template]
            num_total = len(instances)
            num_entail = 0
            sum_probs = 0

            for k in range(0, num_total):
                triple = instances[k]
                sub = triple['head']['word']
                obj = triple['tail']['word']
                tokens = triple['sentence']
                #s = detokenizer.detokenize(tokens).lower()
                rp_tokens = replaceTK(tokens, triple['head'], triple['tail'])
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
                sum_probs += result['probs'][0]

            if num_entail/num_total >= majority_threshold:
                ave_probs = sum_probs/num_total
                net.add_edge(i, j, label=str(format(ave_probs, '.2f')))
                G.add_edge(id2label_dict[i], id2label_dict[j], weight=format(ave_probs, '.2f'))
                edge_weight_matrix[i][j] = format(result['probs'][0], '.2f')

    end = time.time()
    print("used "+str(end-start)+"s")

overall_etime = time.time()

net.show_buttons(filter_=True)

net.show('h_test_test_all2test_tk.html')
nx.write_gml(G, 'h_test_test_all2test_tk.gml')

with open('output/h_test_test_entail_edge_weight_matrix_all2test_tk', 'w') as outF:
    np.savetxt(outF, edge_weight_matrix)

print()
print("Total runtime: "+str((overall_etime-overall_stime)/60)+" minutes" )