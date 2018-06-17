import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def trixify(x):
    newlist = []
    for i in x:
        newmatrix = []
        if len(i) < 8:
            newmatrix += i
            num = 8 - len(i)
            zero = [np.zeros(340,)]*num
            newmatrix += zero
        elif len(i) == 8:
            newmatrix = i
        else:
            newmatrix += i[:8]
            eighth = np.mean(i[8:],axis=0)
            newmatrix += eighth
        sc = StandardScaler()
        matrix = sc.fit_transform(np.column_stack(newmatrix)) # Vectors into matrix
        newlist.append(matrix)
    print('All trixified!')
    return newlist

def open_pickle(x): # Loading features file
    with open(x, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
        return file

embd = open_pickle('FeaturesDictionary.pkl')

def matrixpipeline(x, dict):
    filename = x+'POS.pkl'
    pos = open_pickle(filename)
    matrixname = x+'Matrices.pkl'
    vecs = [[dict.get(m) for m in i] for i in pos] #addressing embeddings' dictionary by word
    matrices = trixify(vecs)
    matrices = [i.reshape(i.shape+(1,)) for i in matrices]
    pickle.dump(matrices, open(matrixname, 'wb'))


matrixpipeline('Test', embd)
