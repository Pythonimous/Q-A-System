import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def insen(x):
    count = 0
    newlist = []
    for i in x:
        sen = np.mean(i, axis=0)
        newlist.append(sen)
        count += 1
        print('В предложение: ' + str(count) + ' / ' + str(len(x)) + ' вопросительных предложений.')
        if count % 10 == 0:
            print(str(count*100/len(x))+' % готово.')
    print('Всё в векторах')
    return newlist

def open_pickle(x):
    with open(x, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
        return file


embd = open_pickle('FeaturesDictionary.pkl')

def vectorpipeline(x, dict):
    filename = x+'POS.pkl'
    pos = open_pickle(filename)
    senvecsname = x+'VectorFeatures.pkl'
    vecs = [[dict.get(m) for m in i] for i in pos] #addressing embeddings' dictionary by word
    sc = StandardScaler()
    senvecs = sc.fit_transform(insen(vecs))
    senvecs = [i.reshape(i.shape+(1,)) for i in senvecs] #step dimension
    pickle.dump(senvecs, open(senvecsname, 'wb'))

vectorpipeline('Test', embd)