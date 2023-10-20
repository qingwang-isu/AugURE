from openie import StanfordOpenIE
import json
import nltk
from nltk import tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def allStops(input_template):
    stop_words = ['the', 'to', 'and', 'a', 'an', 'in', 'it', 'is', 'are', 'of', 'I', 'that',
                  'had', 'on', 'for', 'were', 'was', 'from', 'by', 'with', 'have', 'has', 'be']

    t_tokens = input_template.split(" ")
    result = True
    for tk in t_tokens:
        if tk not in stop_words:
            result = False
            break
    return result

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



def main():
    inF_train = open('../data_sample_for_exemple/tacred_train_sentence.json', 'r')
    inF_val = open('../data_sample_for_exemple/tacred_dev_sentence.json', 'r')
    inF_test = open('../data_sample_for_exemple/tacred_test_sentence.json', 'r')
    inF_chat = open('../data_sample_for_exemple/tacred_train_chatGPT_sentence.json', 'r')

    train_dict = json.load(inF_train)
    val_dict = json.load(inF_val)
    test_dict = json.load(inF_test)
    chat_dict = json.load(inF_chat)
    inF_train.close()
    inF_val.close()
    inF_test.close()
    inF_chat.close()

    freq_threshold = 2
    template_instance_freq = {}
    train_template_instance_dict = {}

    detokenizer = TreebankWordDetokenizer()
    total_cnt = 0

    with StanfordOpenIE() as client:
        print(len(train_dict))
        for instance_train in train_dict:
            if total_cnt % 200 == 0:
                print("finished "+str(total_cnt)+" sentences")

            h_train, t_train, tokens_train = getFormattedItems(instance_train)
            s_train = detokenizer.detokenize(tokens_train)
            try:
                # run OpenIE
                pred_dict_train = {}
                for triple_train in client.annotate(s_train):
                    sub_train = triple_train["subject"].lower()
                    predicate_train = triple_train["relation"].lower()
                    obj_train = triple_train["object"].lower()
                    if allStops(predicate_train):
                        continue
                    # match head & tail with sub & obj
                    if (h_train in sub_train) and (t_train in obj_train):
                        if predicate_train not in pred_dict_train:
                            pred_dict_train[predicate_train] = 0
                            if predicate_train not in template_instance_freq:
                                template_instance_freq[predicate_train] = 1
                            else:
                                template_instance_freq[predicate_train] += 1
                            if predicate_train not in train_template_instance_dict:
                                train_template_instance_dict[predicate_train] = [instance_train]
                            else:
                                train_template_instance_dict[predicate_train].append(instance_train)
            except Exception:
                print(instance_train)
            total_cnt += 1

        print(len(chat_dict))
        total_cnt = 0
        for instance in chat_dict:
            if total_cnt % 200 == 0:
                print("finished " + str(total_cnt) + " sentences")
            h, t, tokens = getFormattedItems(instance)
            s = detokenizer.detokenize(tokens)
            try:
                # run OpenIE
                pred_dict = {}
                for triple in client.annotate(s):
                    sub = triple["subject"].lower()
                    predicate = triple["relation"].lower()
                    obj = triple["object"].lower()
                    if allStops(predicate):
                        continue
                    # match head & tail with sub & obj
                    if (h in sub) and (t in obj):
                        if predicate not in pred_dict:
                            pred_dict[predicate] = 0
                            if predicate not in template_instance_freq:
                                template_instance_freq[predicate] = 1
                            else:
                                template_instance_freq[predicate] += 1
                            if predicate not in train_template_instance_dict:
                                train_template_instance_dict[predicate] = [instance]
                            else:
                                train_template_instance_dict[predicate].append(instance)

            except Exception:
                print(instance)
            total_cnt += 1

        print(len(val_dict))
        total_cnt = 0
        for instance in val_dict:
            if total_cnt % 200 == 0:
                print("finished " + str(total_cnt) + " sentences")
            h, t, tokens = getFormattedItems(instance)
            #tokens = instance["sentence"]
            #h = instance["head"]["word"].lower()
            #t = instance["tail"]["word"].lower()
            s = detokenizer.detokenize(tokens)
            try:
                # run OpenIE
                pred_dict = {}
                for triple in client.annotate(s):
                    sub = triple["subject"].lower()
                    predicate = triple["relation"].lower()
                    obj = triple["object"].lower()
                    if allStops(predicate):
                        continue
                    # match head & tail with sub & obj
                    if (h in sub) and (t in obj):
                        if predicate not in pred_dict:
                            pred_dict[predicate] = 0
                            if predicate not in template_instance_freq:
                                template_instance_freq[predicate] = 1
                            else:
                                template_instance_freq[predicate] += 1
            except Exception:
                print(instance)
            total_cnt += 1

        print(len(test_dict))
        total_cnt = 0
        for instance in test_dict:
            if total_cnt % 200 == 0:
                print("finished "+str(total_cnt)+" sentences")
            h, t, tokens = getFormattedItems(instance)
            s = detokenizer.detokenize(tokens)
            try:
                # run OpenIE
                pred_dict = {}
                for triple in client.annotate(s):
                    sub = triple["subject"].lower()
                    predicate = triple["relation"].lower()
                    obj = triple["object"].lower()
                    if allStops(predicate):
                        continue
                    # match head & tail with sub & obj
                    if (h in sub) and (t in obj):
                        if predicate not in pred_dict:
                            pred_dict[predicate] = 0
                            if predicate not in template_instance_freq:
                                template_instance_freq[predicate] = 1
                            else:
                                template_instance_freq[predicate] += 1
            except Exception:
                print(instance)
            total_cnt += 1



    template_instance_dict_new = {}
    for key in train_template_instance_dict:
        lst = train_template_instance_dict[key]
        if template_instance_freq[key] >= freq_threshold:
            template_instance_dict_new[key] = lst

    json.dump(template_instance_dict_new, open('output/tacred_train_union_template_instance_dict.json', 'w'))


if __name__ == "__main__":
    main()    