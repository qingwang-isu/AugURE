import json
import openai

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
            if type1 == "":
                type1 += tk.lstrip("<e1:").rstrip(">")
                in1 = True
        elif "</e1:" in tk:
            in1 = False
        elif "<e2:" in tk:
            if type2 == "":
                type2 += tk.lstrip("<e2:").rstrip(">")
                in2 = True
        elif "</e2:" in tk:
            in2 = False

        if ("<e1:" not in tk) and (in1 == True):
            if tk == "'s":
                e1 = e1.rstrip(" ")
            e1 += tk
            e1 += " "
        if ("<e2:" not in tk) and (in2 == True):
            if tk == "'s":
                e2 = e2.rstrip(" ")
            e2 += tk
            e2 += " "

    out_e1 = e1.rstrip(" ")
    out_e2 = e2.rstrip(" ")

    spe_lst = ["it", "its", "he", "his", "him", "she", "her", "they", "their", "them", "I"]
    if out_e1.lower() in spe_lst:
        out_e1 = '"' + out_e1 + '"'
    if out_e2.lower() in spe_lst:
        out_e2 = '"' + out_e2 + '"'
    return type1, out_e1, type2, out_e2


def removeMarkers(sen, e1, e2):
    result = ""
    tokens = sen.split(" ")
    ori_e1 = ""
    ori_e2 = ""
    e1_start = False
    e2_start = False
    for i in range(len(tokens)):
        tk = tokens[i]
        if "<e1:" in tk:
            e1_start = True
            continue
        elif "</e1:" in tk:
            if ori_e1.rstrip(" ") != e1:
                result += e1
                result += " "
            else:
                result += ori_e1
            e1_start = False
            continue
        elif "<e2:" in tk:
            e2_start = True
            continue
        elif "</e2:" in tk:
            if ori_e2.rstrip(" ") != e2:
                result += e2
                result += " "
            else:
                result += ori_e2
            e2_start = False
            continue
        else:
            if e1_start == False and e2_start == False:
                result += tk
                result += " "
            elif e1_start == True:
                ori_e1 += tk
                ori_e1 += " "
            elif e2_start == True:
                ori_e2 += tk
                ori_e2 += " "
    return result.rstrip(" ")


def capitalize_nth(s, n):
    return s[:n].lower() + s[n:].capitalize()


def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s


def addMarkers(rp_sen, type1, e1, type2, e2):
    rp_e1 = "<e1:" + type1 + "> " + e1 + " </e1:" + type1 + ">"
    rp_e2 = "<e2:" + type2 + "> " + e2 + " </e2:" + type2 + ">"
    result_sen1 = rp_sen.replace(e1, rp_e1, 1)
    result_sen1 = result_sen1.replace(e2, rp_e2, 1)

    if "<e1" in result_sen1 and "<e2" in result_sen1:
        return result_sen1
    else:
        # reverse order
        result_sen2 = rp_sen.replace(e2, rp_e2, 1)
        result_sen2 = nth_repl(result_sen2, e1, rp_e1, 2)
        if "<e1" in result_sen2 and "<e2" in result_sen2:
            return result_sen2

    result_sen2 = rp_sen.replace(e2, rp_e2, 1)
    result_sen2 = result_sen2.replace(e1, rp_e1, 1)
    spe_lst = ['"it"', '"its"', '"he"', '"his"', '"him"', '"she"', '"her"', '"they"', '"their"', '"them"']
    # check capitalized entities
    if "<e1" not in result_sen2 and "<e2" in result_sen2:
        if e1 in spe_lst:
            tmp_e1 = capitalize_nth(e1, 1)
            tmp_rp_e1 = "<e1:" + type1 + "> " + tmp_e1 + " </e1:" + type1 + ">"
            result_sen3 = result_sen2.replace(tmp_e1, tmp_rp_e1, 1)
            if "<e1" in result_sen3:
                return result_sen3
        elif e1.capitalize() in rp_sen:
            tmp_e1 = e1.capitalize()
            tmp_rp_e1 = "<e1:" + type1 + "> " + tmp_e1 + " </e1:" + type1 + ">"
            result_sen3 = result_sen2.replace(tmp_e1, tmp_rp_e1, 1)
            if "<e1" in result_sen3:
                return result_sen3
    elif "<e2" not in result_sen1 and "<e1" in result_sen1:
        if e2 in spe_lst:
            tmp_e2 = capitalize_nth(e2, 1)
            tmp_rp_e2 = "<e2:" + type2 + "> " + tmp_e2 + " </e2:" + type2 + ">"
            result_sen4 = result_sen1.replace(tmp_e2, tmp_rp_e2, 1)
            if "<e2" in result_sen4:
                return result_sen4
        elif e2.capitalize() in rp_sen:
            tmp_e2 = e2.capitalize()
            tmp_rp_e2 = "<e2:" + type2 + "> " + tmp_e2 + " </e2:" + type2 + ">"
            result_sen4 = result_sen1.replace(tmp_e2, tmp_rp_e2, 1)
            if "<e2" in result_sen4:
                return result_sen4
    if e1 not in rp_sen and e2 not in rp_sen:
        return rp_e1 + " and " + rp_e2 + " are " + rp_sen
    return result_sen1


inF = open("../data_sample_for_exemple/tacred_train_sentence.json", "r")
input_lst = json.load(inF)
print("total # sentences")
print(len(input_lst))

# need to add your own OpenAI api key here!!!
openai.api_key = ''
outF = open("train_chatGPT_sentence_him.json", "w")

reply_messages = []
no_rel = 0
for i in range(0, len(input_lst)):
    if i % 100 == 0:
        print(i)
    s = input_lst[i]
    t1, ent1, t2, ent2 = getTypedEntities(s)
    rm_s = removeMarkers(s, ent1, ent2)
    if ent1 == '"him"' or ent1 == '"them"' or ent2 == '"him"' or ent2 == '"them"':
        q_sen = 'Given the context "' + rm_s + '", what is the relationship between ' + ent1 + ' and ' + ent2 + ' (as short as possible)?'
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",  # this is "ChatGPT" $0.002 per 1k tokens
        messages=[{"role": "user", "content": q_sen}])
        reply = completion.choices[0].message.content
        m_reply = addMarkers(reply, t1, ent1, t2, ent2)

        outF.write(str(i))
        outF.write("\n")
        outF.write(m_reply)
        outF.write("\n")
        reply_messages.append(m_reply)


with open("../data_sample_for_exemple/tacred_train_chatGPT_sentence.json", "w") as f:
    json.dump(reply_messages, f)