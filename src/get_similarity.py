from cmath import cos
import numpy as np
import string
import pandas as pd
from requests import get
from scipy.spatial.distance import cosine
import joblib
from stopwords_process import remove_stopwords
import random
'''
def get_glove_emebdding(embedding_path):
    f = open(embedding_path, 'r')
    lines = f.readlines()
    dic = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        else:
            line = line.strip().split()
            if line[0] not in dic:
                dic[line[0]] = [float(w) for w in line[1:]]
    
    return dic
'''
# 把一些实体类型随机mask掉
def random_mask(sentence, ratio=0):
    # sentence = sentence.split()
    length = len(sentence)
    entity_idx = []
    for i, w in enumerate(sentence):
        if w != 'O':
            entity_idx.append(i)
    random.shuffle(entity_idx)
    for i in range(int(ratio*len(entity_idx))):
        sentence[entity_idx[i]] = 'O'
    
    return sentence

def get_sim(sentence):
    length = len(sentence)
    sim_matrix = []
    for i, w in enumerate(sentence):
        # sim_matrix = []
        if i == 0: # 如果是[CLS],则和所有的实体做attention
            for j, w1 in enumerate(sentence):
                if j == i:
                    sim_matrix.append(1)
                else:
                    if w1 != 1: # 1 stands O
                        sim_matrix.append(1)
                    else:
                        sim_matrix.append(0)
        else:
            if w == 1:
                sim_matrix.extend([0]*length)
            else:
                for j, w1 in enumerate(sentence):
                    if j == 0:
                        sim_matrix.append(1)
                    else:
                        if w == w1:
                            sim_matrix.append(1)
                        else:
                            sim_matrix.append(0)
        # sim_matrix.append(sim)

    return sim_matrix
  
def get_cls_unsimilarity(sentence):
    length = len(sentence)
    sim_matrix = []
    cls_w = sentence[0]
    for i, w in enumerate(sentence):
        if w == cls_w:
            sim_matrix.append(1)
        else:
            if w == 1: # 1 stands O
                sim_matrix.append(1)
            else:
                sim_matrix.append(0)

    return sim_matrix
  

class external_similarity(object):
    def __init__(self, embedding_path):
        self.embedding_path = embedding_path
        self.dic = self.get_glove_emebdding()
    
    def get_glove_emebdding(self):
        f = open(self.embedding_path, 'r')
        lines = f.readlines()
        dic = {}
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                line = line.strip().split()
                if line[0] not in dic:
                    dic[line[0]] = [float(w) for w in line[1:]]
        return dic

    def get_sim(self, sentence):
        length = len(sentence)
        sim_matrix = []
        for i, w in enumerate(sentence):
            # sim_matrix = []
            for j, w1 in enumerate(sentence):
                if w == w1:
                    sim_matrix.append(1)
                else:
                    sim_matrix.append(0)
            # sim_matrix.append(sim)

        return sim_matrix

    def get_semantic_glove_sim(self, sentence):
        
        length = len(sentence)
        sim_matrix = []
        new_w = ''
        new_w_k = ''
        for i, w in enumerate(sentence):
            if w.startswith('##'):
                before_w = ''
                after_w = ''
                j1 = i
                j2 = i
                while(j1):
                    if sentence[j1].startswith('##'):
                        before_w = ''.join(sentence[j1][2:]) + before_w
                        j1 = j1 - 1
                    else:
                        break
                while(j2+1<len(sentence)):
                    if sentence[j2+1].startswith('##'):
                        after_w = after_w + ''.join(sentence[j2+1][2:])
                        j2 = j2 + 1
                    else:
                        break
                new_w = before_w + after_w
            else:
                new_w = w
            for k, w1 in enumerate(sentence):
                if w1.startswith('##'):
                    before_w_k = ''
                    after_w_k = ''
                    l1 = k
                    l2 = k
                    while(l1):
                        if sentence[l1].startswith('##'):
                            before_w_k = ''.join(sentence[l1][2:]) + before_w_k
                            l1 = l1 - 1
                        else:
                            break
                    while(l2+1<len(sentence)):
                        if sentence[l2+1].startswith('##'):
                            after_w_k = after_w_k + ''.join(sentence[l2+1][2:])
                            l2 = l2 + 1
                        else:
                            break
                    new_w_k = before_w_k + after_w_k
                else:
                    new_w_k = w1
                # print(new_w)
                # print(new_w_k)
                if new_w in self.dic:
                    emb_w = np.array(self.dic[new_w])
                else:
                    emb_w = np.zeros(200)
                # print(emb_w)
                if new_w_k in self.dic:
                    emb_w_k = np.array(self.dic[new_w_k])
                else:
                    emb_w_k = np.zeros(200)
                # print(emb_w_k)
                if (new_w not in self.dic) or (new_w_k not in self.dic):
                    sim_matrix.append(0.0)
                else:
                    sim_matrix.append(1-cosine(emb_w, emb_w_k))
        return sim_matrix

# dic = get_glove_emebdding('/disk2/xy_disk2/TOOLS/embedding/mimic-NOTEEVENTS200d.glove')
def get_semantic_glove_sim(sentence, dic):
    sentence = sentence.split()
    length = len(sentence)
    sim_matrix = []
    new_w = ''
    new_w_k = ''
    for i, w in enumerate(sentence):
        if w.startswith('##'):
            before_w = ''
            after_w = ''
            j1 = i
            j2 = i
            while(j1):
                if sentence[j1].startswith('##'):
                    before_w = ''.join(sentence[j1][2:]) + before_w
                    j1 = j1 - 1
                else:
                    break
            while(j2+1<len(sentence)):
                if sentence[j2+1].startswith('##'):
                    after_w = after_w + ''.join(sentence[j2+1][2:])
                    j2 = j2 + 1
                else:
                    break
            new_w = before_w + after_w
        else:
            new_w = w
        for k, w1 in enumerate(sentence):
            if w1.startswith('##'):
                before_w_k = ''
                after_w_k = ''
                l1 = k
                l2 = k
                while(l1):
                    if sentence[l1].startswith('##'):
                        before_w_k = ''.join(sentence[l1][2:]) + before_w_k
                        l1 = l1 - 1
                    else:
                        break
                while(l2+1<len(sentence)):
                    if sentence[l2+1].startswith('##'):
                        after_w_k = after_w_k + ''.join(sentence[l2+1][2:])
                        l2 = l2 + 1
                    else:
                        break
                new_w_k = before_w_k + after_w_k
            else:
                new_w_k = w1
            # print(new_w)
            # print(new_w_k)
            if new_w in dic:
                emb_w = np.array(dic[new_w])
            else:
                emb_w = np.zeros(200)
            # print(emb_w)
            if new_w_k in dic:
                emb_w_k = np.array(dic[new_w_k])
            else:
                emb_w_k = np.zeros(200)
            # print(emb_w_k)
            if new_w not in dic and new_w_k not in dic:
                sim_matrix.append(0.0)
            else:
                sim_matrix.append(1-cosine(emb_w, emb_w_k))
    return sim_matrix

import tokenization

def read_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

# vocab_file = '/disk4/xy/PROJECT/2019AGAC/bert_model/uncased_L-12_H-768_A-12/vocab.txt'
# tokenizer = tokenization.FullTokenizer(
#       vocab_file=vocab_file, do_lower_case=True)
# sim = external_similarity('/disk2/xy_disk2/TOOLS/embedding/mimic-NOTEEVENTS200d.glove')

# def cal_sim(filename, tokenizer, max_len, sim, out_file):
#     lines = read_lines(filename)
#     all_sim = []
#     for i, line in enumerate(lines):
#         # print(i)
#         tokens = []
#         tokens.append('[CLS]')
#         line = line.strip().split('\t')
#         _, s1, s2 = line
#         s1 = remove_stopwords(s1)
#         s2 = remove_stopwords(s2)
#         for i, token in enumerate(s1.split()):
#             tos = tokenizer.tokenize(token)
#             for t in tos:
#                 tokens.append(t)
#         tokens.append('[SEP]')
        
#         for i, token in enumerate(s2.split()):
#             tos = tokenizer.tokenize(token)
#             for t in tos:
#                 tokens.append(t)
        
#         if len(tokens) > max_len - 1:
#             tokens = tokens[max_len-1]
#         tokens.append('[SEP]')
#         sim_matrix = sim.get_semantic_glove_sim(tokens)
#         all_sim.append(sim_matrix)
#     joblib.dump(all_sim, out_file)

# outfile3 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/test_data/test_sim_stopwords.pkl'
# input3 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/test_data/test.out'
# for i in range(5):
#     input1 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/data_' + str(i+1) + '/dev.out'
#     input2 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/data_' + str(i+1) + '/train.out'

#     outfile1 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/data_' + str(i+1) + '/dev_sim_stopwords.pkl'
#     outfile2 = '/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/data_' + str(i+1) + '/train_sim_stopwords.pkl'

#     cal_sim(input1, tokenizer, 300, sim, outfile1)
#     cal_sim(input2, tokenizer, 300, sim, outfile2)
# cal_sim(input3, tokenizer, 300, sim, outfile3)
# x = joblib.load('/home/xy/xy_pro/xy_disk2/2019n2c2-copy/processed_data/test_data/test_sim.pkl')
# print(x[0])
# sentence = 'O O O O O O O O SignSymptomMention AnatomicalSiteMention AnatomicalSiteMention ProcedureMention'
# sen = random_mask(sentence, 0.8)
# print(sen)
