import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib


def read_lines(filename):
    f = open(filename, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    return lines

def evaluate(file1, file2):
    lines = read_lines(file1)
    gold = []
    for line in lines:
        tmp = int(line.strip('\n').split('\t')[0])
        gold.append(tmp)
    result = joblib.load(file2)
    print(len(result))
    # print(result[:100])
    f = open('BERT_result_dev.txt', 'w', encoding='utf8')
    for r in result:
        f.write(str(r) + '\n')
    p = precision_score(gold, result, average='micro')
    r = recall_score(gold, result, average='micro')
    f = f1_score(gold, result, average='micro')
    print('p, r, f:', p, r, f)

if __name__ == "__main__":
    evaluate('/raid/xy/PROJECT/PharmaCoNER_data/bert_data_v2/dev.out', 'bert_result_test.pkl')
