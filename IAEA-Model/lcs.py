import numpy as np
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd
def find_lcseque(s1, s2 ,former_sent, current_sent):
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    print(s1)
    print(s2)
    # print(former_sent)
    # print(current_sent)
    # s1 = np.array(s1)
    # s2 = np.array(s2)
    # ner_s1, index1 = ner_detect(former_sent)
    # ner_s2, index2 = ner_detect(current_sent)
    # print("s1 ", type(s1))
    # if ner_s1 == 2:
    #     s1.append("ORGANIZATION")
    #     s1.append("CD")
    # elif ner_s1 == 1:
    #     s1.append("CD")
    # elif ner_s1 == 0:
    #     s1.append("ORGANIZATION")
    # if ner_s2 == 2:
    #     s2.append("ORGANIZATION")
    #     s2.append("CD")
    # elif ner_s2 == 1:
    #     s2.append("CD")
    # elif ner_s2 == 0:
    #     s2.append("ORGANIZATION")
    # print(s1)
    # print(s2)
    # new_s1 = []
    # new_s2 = []
    # for i in len(s1):
    #     if i not in index1:
    #         new_s1.append(s1[i])
    # for i in len(s2):
    #      if i not in index1:
    #          new_s2.append(s2[i])

    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2]
                d[p1+1][p2+1] = 'left'
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]
                d[p1+1][p2+1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    #print (numpy.array(d))
    s = []
    while m[p1][p2]:    #不为None时
        c = d[p1][p2]
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1-=1
            p2-=1
        if c =='left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    # print((''.join(s)))
    print(s)
    return len(s)

def ner_detect(article_sent):
    tokenized_sentence = nltk.word_tokenize(article_sent)
    #print(tokenized_sentence)
    # tag sentences and use nltk's Named Entity Chunker
    tagged_sentence = nltk.pos_tag(tokenized_sentence)
    #print(tagged_sentence)
    ne_chunked_sent = nltk.ne_chunk(tagged_sentence)
    # print(type(ne_chunked_sent))
    # extract all named entities
    # named_entities = []
    index = []
    flag_org = 0
    flag_cd = 0
    i = 0
    for tagged_tree in ne_chunked_sent:
        # extract only chunks having NE labels
        # print(type(tagged_tree))
        if "ORGANIZATION" in str(tagged_tree) or "GPE" in str(tagged_tree):
            # print(tagged_tree)
            flag_org += 1
            index.append(i)
        if "CD" in str(tagged_tree):
            # print(tagged_tree)
            flag_cd += 1
            index.append(i)
        i += 1
    print(index)
    # print(flag_org)
    # print(flag_cd)
    if (flag_org != 0 and flag_cd != 0):
        return 2, index
    elif flag_org != 0:
        return 0, index
    elif flag_cd != 0:
        return 1, index
    else:
        return -1, index



def simplex_inconsistent_detect(article_sents):
    former_sents = article_sents[0:len(article_sents)-1]
    current_sent = article_sents[-1]
    words = stopwords.words('english')
    # print(words)
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
        words.append(w)
    disease_List = nltk.word_tokenize(current_sent)
    filtered_current_words = [word for word in disease_List if word not in words]
    for former_sent in former_sents:
        disease_List = nltk.word_tokenize(former_sent)
        filtered_fromer_words = [word for word in disease_List if word not in words]
        lcs_len = find_lcseque(filtered_current_words, filtered_fromer_words, former_sent, current_sent)
        print(lcs_len)
        print(len(filtered_current_words))
        print(len(filtered_fromer_words))
        if lcs_len > max(len(filtered_current_words), len(filtered_fromer_words)) * 0.5:
            print("#######################################")
            return True
    return False

str1 = "#bombs took all day 4 officials 2 admit their were bombs deadly bombs rock boston marathon - terrorattck"
str2 = "3 killed , more than 140 injured in boston marathon bombing"
str3 = "boston pd says at least 3 people have died from boston marathon explosions"
# ner_detect(str1)
article_sents = []
article_sents.append(str1)
article_sents.append(str2)
article_sents.append(str3)
article_sents = np.array(article_sents)
simplex_inconsistent_detect(article_sents)