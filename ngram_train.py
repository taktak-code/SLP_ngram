import collections as co
import pickle
import sys
import math

class Laplace():
    def __init__(self, n): #モデル: n-1gramとngramと語彙サイズ
        self.n = n
        self.dic_n_1, self.dic_n = dict(), dict()
        self.V = 0
    
    def ngram(self, k, text): # 入力: (2, "私　は　タコ　が　好き　で　す /s")　出力: ["<s> 私", "私 は", ...]
        l = []
        if self.n >= 2:
            words=['<s>']*(self.n-1)
        else: #n==1の場合にも<s>を追加したい
            words=['<s>']
        words+=text.split()
        for i in range(0, len(words)-k+1): #内包表記が良いかも
            l.append(' '.join(words[i:i+k])) #空白区切りで連結
        return l

    def train(self): #1行ずつ読み込んで辞書とVを更新
        count_n1g=co.Counter()
        count_ng=co.Counter()
        voc_set=set() #Vはsetを使ってuniqueなunigramを集める
        for text in sys.stdin:
            count_n1g.update(self.ngram(self.n-1, text))
            count_ng.update(self.ngram(self.n, text))
            for token in text.split():
                    voc_set.add(token)
        self.dic_n_1 = dict(count_n1g)
        self.dic_n = dict(count_ng)
        self.V = len(voc_set)+1 #<UNK>の分+1する

N = 2 #bigram
f_name='bigram.pickle'

if __name__ == "__main__":
    ng = Laplace(N)
    ng.train()
    with open(f_name, 'wb') as f:
        pickle.dump(ng, f)