import numpy as np
import pickle
import tensorflow as tf

sess = tf.Session()

def tensorify(x):
    count = 0
    newlist = []
    for i in x:
        newmatrix = []
        if len(i) < 8:
            newmatrix += i
            num = 8 - len(i)
            zero = [np.zeros(340, )] * num
            newmatrix += zero
        elif len(i) == 8:
            newmatrix = i
        else:
            newmatrix += i[:8]
            eighth = np.mean(i[8:], axis=0)
            newmatrix += eighth

        trimatrices = []
        for d in range(2, len(newmatrix)):
            matrix = np.column_stack([newmatrix[d-2], newmatrix[d-1], newmatrix[d]])
            trimatrices.append(matrix)
        tens2 = np.dstack(trimatrices)
        tens1 = tf.nn.l2_normalize(tens2) # L2-нормализация, но приобретает формат tf.tensor
        tens = sess.run(tens1) # Возвращаем к формату np.array2
        newlist.append(tens)
        count += 1
        print('Тензорировано: ' + str(count) + ' / ' + str(len(x)) + ' вопросительных предложений,'
                                                                     ' форма последнего тензора '+str(tens.shape))
        if count % 10 == 0:
            print(str(count*100/len(x))+' % готово.')
    print('Всё тензоризировано.')
    return newlist

def open_pickle(x):
    with open(x, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
        return file

embd = open_pickle('FeaturesDictionary.pkl')

def tensorpipeline(x, dict):
    filename = x+'POS.pkl'
    pos = open_pickle(filename)
    tensorname = x+'Tensors.pkl'
    vecs = [[dict.get(m) for m in i] for i in pos] #addressing embeddings' dictionary by word
    tensors = tensorify(vecs)
    tensors = [i.reshape(i.shape+(1,)) for i in tensors] # step dimension
    pickle.dump(tensors, open(tensorname, 'wb'))

tensorpipeline('Training', embd)