import os
import numpy as np

def read_lines(filename):
    with open(filename, 'r', encoding='utf8') as f:
        return f.readlines()

def write_lines(filename, result):
    with open(filename, 'w', encoding='utf8') as f:
        for r in result:
            f.write(str(r) + '\n')

# root_dir = '/userhome/xy_pro/STS/src/N2C2_fix'
# fold = 5
# out_file = os.path.join(root_dir, 'vote_n2c2_bert_entityemb_entityattention_01.txt')
# gold_file = os.path.join('/userhome/xy_pro/STS/data/2019n2c2/', 'test_data/test_kg.txt')
# filelist = ['n2c2_bert_entityemb_entityattention_all_1/test_result.txt', 'n2c2_bert_entityemb_entityattention_all_2/test_result.txt', 'n2c2_bert_entityemb_entityattention_all_3/test_result.txt', 'n2c2_bert_entityemb_entityattention_all_4/test_result.txt', 'n2c2_bert_entityemb_entityattention_all_5/test_result.txt']

# root_dir = '/disk2/xy_disk2/thesis/STS/LAP/src/bert_src/'
# fold = 5
# out_file = os.path.join(root_dir, 'vote_ebmsass_retrieval.txt')
# gold_file = os.path.join("/disk2/xy_disk2/thesis/STS/LAP/data/EBMSASS/", 'all_kg.txt')
# filelist = ['n2c2_biobert_retrival_5/test_result.txt', 'n2c2_biobert_retrival_1/test_result.txt', 'n2c2_biobert_retrival_2/test_result.txt', 'n2c2_biobert_retrival_3/test_result.txt', 'n2c2_biobert_retrival_4/test_result.txt']



root_dir = '/disk2/xy_disk2/thesis/STS/LAP/src/bert_src/'
fold = 5
out_file = os.path.join(root_dir, 'vote_biosses_retrieval.txt')
gold_file = os.path.join("/disk2/xy_disk2/thesis/STS/LAP/data/EBMSASS/", 'all_kg.txt')
filelist = ['EBMSASS_biobert_new_5/test_result.txt', 'EBMSASS_biobert_new_1/test_result.txt', 'EBMSASS_biobert_new_2/test_result.txt', 'EBMSASS_biobert_new_3/test_result.txt', 'EBMSASS_biobert_new_4/test_result.txt']


def vote(filelist, out_file):
    results = []
    out_result = []
    for file in filelist:
        filename = os.path.join(root_dir, file)
        results.append(read_lines(filename)[:-2])
    lines = len(results[0])
    print(lines)
    for i in range(lines):
        score = 0.0
        for j in range(fold):
            # if j == 4:
            #     continue
            score += float(results[j][i])
        out_result.append(score/len(filelist))
                
    write_lines(out_file, out_result)
    
    gold_lines = read_lines(gold_file)[800:]
    print(len(gold_lines))
    gold = []
    for line in gold_lines:
#         print(line)
        tmp = float(line.strip('\n').split('\t')[-1])
        gold.append(tmp)
    test_pearson = np.corrcoef(np.array(out_result),np.array(gold))[0][1]
    print('皮尔逊系数为: ',test_pearson)

def avg_pear(filelist):
    results = 0.0
    for i, file in enumerate(filelist):
        # if i==7:
        #     continue
        filename = os.path.join(root_dir, file)
        # print(filename)
        score = float(read_lines(filename)[-2].split(':')[-1])
        results = results + score
        # print(results)
    # print(results)
    results = results/len(filelist)
    # print(results)
    return results

def avg_spear(filelist):
    results = 0.0
    for i, file in enumerate(filelist):
        # if i == 7:
        #     continue
        filename = os.path.join(root_dir, file)
        score = float(read_lines(filename)[-1].split(':')[-1])
        results += score
                
    results = results/len(filelist)
    return results

vote(filelist, out_file)
   
new_fold = 5
filelist = []
# model_dir = '/userhome/xy_pro/STS/src/BIOSESS_final/biosess_bert_'
# model_dir = '/userhome/xy_pro/STS/src/BIOSESS_final/biosess_bert_entityemb_'
# model_dir = '/userhome/xy_pro/STS/src/BIOSESS_adp/biosess_roberta_raw_newnew_'
# model_dir = '/userhome/xy_pro/STS/src/EBMSASS_adp/ebmsass_roberta_entityemb_entityatt_new_'
# root_dir = '/userhome/xy_pro/STS/src/BIOSESS_adp'
# out_file = os.path.join(root_dir, 'vote_biosess_roberta_raw_newnew.txt')
# gold_file = os.path.join('/userhome/xy_pro/STS/data/BIOSSES/', 'all_kg.txt')
# root_dir = '/userhome/xy_pro/STS/src/EBMSASS_adp'
# out_file = os.path.join(root_dir, 'vote_ebmsass_roberta_entityemb_entityatt_03.txt')
# gold_file = os.path.join('/userhome/xy_pro/STS/data/EBMSASS/', 'all_kg.txt')

# model_dir = '/userhome/xy_pro/STS/src/EBMSASS_new/BIOSESS_bert_re_entityemb_'
# for i in range(new_fold):
#     if i == 3:
#      continue
    # filelist.append(os.path.join(model_dir+str(i+1), 'test_result.txt'))
# vote(filelist, out_file)

# avg = avg_pear(filelist)
# print(avg)
# spear = avg_spear(filelist)
# print(spear)
