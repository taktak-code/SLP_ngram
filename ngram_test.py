import math
import sys
import pickle
from ngram_train import Laplace

def calc_prob(text, model): #text(一文)の生成確率を出力
    x = model.ngram(model.n, text)
    logP = 0
    N = 0
    for i in x:
        N += 1
        if i in model.dic_n:
            j = ' '.join(i.split()[:-1]) #対応するn-1gramの文字列, c(Wのn-N+1~n-1)に利用
            logP += math.log((model.dic_n[i]+1)/(model.dic_n_1[j]+model.V)) #logで桁落ちを抑える
        else:
            logP += math.log(1/model.V)
    return logP, N

def calc_pp(model): #テストデータ全体のパープレキシティ
    logP, N = 0, 0
    for text in sys.stdin:
        x, y = calc_prob(text, model)
        logP += x
        N += y
    PP = math.exp(-1*logP/N)
    return PP

if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        model = pickle.load(f)

    print('Perplexity: %f'%calc_pp(model))
    print('n-1gram size: %d'%len(model.dic_n_1))
    print('ngram size: %d'%len(model.dic_n))
    #calc_probでiとlogPを返す実装にして、この2つをひたすら加算して、最後にprobに戻して-1/i乗する