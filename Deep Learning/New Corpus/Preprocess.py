import re
import gensim
import numpy as np
import pymorphy2
import string
import pickle
import tensorflow as tf
import timeit
from pymystem3 import Mystem

sess = tf.Session()
model = gensim.models.KeyedVectors.load_word2vec_format('ruscorpora_upos_skipgram_300_10_2017.bin.gz',
                                                        binary=True)  # pre-trained vector model for extracting features
morph = pymorphy2.MorphAnalyzer()
m = Mystem()

def extract_questions(x): #x - labeled data name, here separator is \t
    lines = [line.rstrip('\n') for line in open(x, encoding='utf-8-sig')]
    q = []
    for i in lines:
        i = i.split('\t')
        q.append(i[0])
    print('Question extraction... done!')
    return q

def extract_labels(x):
    lines = [line.rstrip('\n') for line in open(x, encoding='utf-8-sig')]
    q = []
    for i in lines:
        i = i.split('\t')
        q.append(i[1])
    print('Label extraction... done!')
    return q

def preprocformatrices(x): # preprocessing for matrices
    list2 = extract_questions(x)
    list1 = []
    for i in list2:
        i = i.rstrip().lower().replace('кое-','кое- ').replace('-то',' -то').replace('-либо',' -либо').\
            replace('-нибудь',' -нибудь').replace('-таки',' -таки').replace('ё','е')
        i = ''.join(ch for ch in i if ch not in set(string.punctuation)).replace('»', '').replace('«', '').replace('—', '').replace('–', '')
        i = ' '.join(i.split()).split(' ')
        list1.append(i)

  #  avg_len = round(sum([len(i) for i in list1])/len(list1)) + 1
    avg_len = 8
    print('Average sentence length: '+str(avg_len))
    list = []
    for i in list1:
        if len(i) == avg_len:
            new = i
        elif len(i) > avg_len:
            new = i[:avg_len]
        else:
            num_empty = avg_len - len(i)
            empties = ['и']*num_empty
            new = i + empties
        list.append(new)

    return list

def POS_tagger(i): #for word2vec single sentence / word processing

    sen = m.lemmatize(i) #sentence to list of lemmas

    sen1 = sen[:-1]

    for a in sen1:
        if a == ' ':
            sen1.remove(a)

    sent = []
    for a in sen1:
        if a == 'и':
            b = 'и_CONJ'
        else:
            b = a + '_' + str((morph.parse(a)[0]).tag.POS)
        sent.append(b)

    return sent

def vectorize(x): #presenting word as vector

    sent1 = []
    for s in x:
        sent2 = []
        for i in s:
            if i in model:
                vector = model[i]
                sent2.append(vector)
            else:
                continue
        sent1.append(sent2)

    sent3 = []
    for i in sent1:
        if len(i) == 0: #if word has not been found in a pre-trained model
            i = np.zeros((300,))
            sent3.append(i)
        else:
            sent3.append(np.mean(i, axis=0)) #one-vector word representation
    return sent3

def determinefeatures(a): #extracting additional features from a word

    if a != 'и':
        list = []
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0
        t6 = 0
        t7 = 0
        t8 = 0
        t9 = 0
        t10 = 0
        t11 = 0
        t12 = 0
        t13 = 0
        t14 = 0
        t15 = 0
        t16 = 0
        t17 = 0
        t18 = 0
        t19 = 0
        t20 = 0
        t21 = 0
        t22 = 0
        t23 = 0
        t24 = 0
        t25 = 0
        t26 = 0
        t27 = 0
        t28 = 0
        t29 = 0
        t30 = 0
        t31 = 0
        t32 = 0
        t33 = 0
        t34 = 0
        t35 = 0
        t36 = 0
        t37 = 0
        t38 = 0
        t39 = 0
        t40 = 0

        if a in ['что', 'кто', 'кого','чего','кому','чему','чем','кем']:
            t1 = 1
        list.append(t1)
        if a == 'то':
            t2 = 1
        list.append(t2)
        if a in ['означает', 'такое', 'значит']:
            t3 = 1
        list.append(t3)
        if a == 'ли':
            t4 = 1
        list.append(t4)
        if a in ['пример', 'например']:
            t5 = 1
        list.append(t5)
        if a == 'где':
            t6 = 1
        list.append(t6)
        if a == 'куда':
            t7 = 1
        list.append(t7)
        if a == 'когда':
            t8 = 1
        list.append(t8)
        if a == 'откуда':
            t9 = 1
        list.append(t9)
        if a == 'почему':
            t10 = 1
        list.append(t10)
        if a == 'зачем':
            t11 = 1
        list.append(t11)
        if a == 'как':
            t12 = 1
        list.append(t12)
        if a == 'чем':
            t13 = 1
        list.append(t13)
        if 'похож' in a:
            t14 = 1
        list.append(t14)
        if a in ['отличается','отличаются']:
            t15 = 1
        list.append(t15)
        if a == 'чем':
            t16 = 1
        list.append(t16)
        if 'разниц' or 'отлич' in a:
            t17 = 1
        list.append(t17)
        if a == 'или':
            t18 = 1
        list.append(t18)
        if a == 'если':
            t19 = 1
        list.append(t19)
        if a in ['за','после','дальше','далее']:
            t20 = 1
        list.append(t20)
        if 'цел' in a or 'задач' in a:
            t21 = 1
        list.append(t21)
        if a in ['было','явилось','стало','послужило']:
            t22 = 1
        list.append(t22)
        if a in ['дальше','следует','будет'] or 'последств' in a or a == 'из':
            t23 = 1
        list.append(t23)
        if 'причин' in a:
            t24 = 1
        list.append(t24)
        if 'скольк' in a:
            t25 = 1
        list.append(t25)
        if a == 'не':
            t26 = 1
        list.append(t26)
        if 'мног' in a:
            t27 = 1
        list.append(t27)
        if re.match(r'как(ой|ая|ое|ие|ого|ому|им|ом|ую|их|ими)', a):
            t28 = 1
        list.append(t28)
        if 'свойств' or 'качеств' in a:
            t29 = 1
        list.append(t29)
        if 'год' or 'месяц' or 'минут' or 'час' or 'секунд' or 'числ' in a or a in ['день','дня','дню','днем','дне']:
            t30 = 1
        list.append(t30)
        if a == 'через':
            t31 = 1
        list.append(t31)
        if a in {'в', 'во'}:
            t32 = 1
        list.append(t32)
        if re.match(r'котор(ый|ая|ое|ые|ого|ому|ым|ом|ую|ых|ыми|ой)', a):
            t33 = 1
        list.append(t33)
        if 'каков' in a:
            t34 = 1
        list.append(t34)
        if 'помощ' or 'посредств' in a:
            t35 = 1
        list.append(t35)
        for i in ['схем','тактик','алгоритм','образ','план']:
            if i in a:
                t36 = 1
        if re.match(r'пут(ь|и|ем|ей|ям|ями|ях)', a):
            t36 = 1
        list.append(t36)
        if a == 'чтобы':
            t37 = 1
        list.append(t37)
        if re.match(r'наделен([аоы]?)', a) or re.match(r'облада[ею]т', a) or re.match(r'характеризу[ею]тся', a) or re.match(r'име[ею]т', a):
            t38 = 1
        list.append(t38)
        m = re.match(r'воспользова(лся|лась|ться)|благодаря|позволил(о?)|образом', a)
        if m and (t5 != 1):
            t39 = 1
        list.append(t39)
        if a in ['же', 'ж']:
            t40 = 1
        list.append(t40)

    else:
        list = [0]*40

    return list

def vecsnfeatures(x): #adding regexp features as a binary sequence to the end of the word array
    vectors = vectorize(POS_tagger(x))
    features = determinefeatures(x)
    y = np.append(vectors, features)
    return y

def trixify(x):
    count = 0
    newlist = []
    for i in x:
        trimatrix = []
        for d in i:
            trimatrix.append(vecsnfeatures(d)) #Turning words into vectors
        matrix = np.column_stack(trimatrix) # Vectors into matrix
        newlist.append(matrix)
        count += 1
        print('Into matrices: ' + str(count) + ' / ' + str(len(x)) + ' questions,'
                                                                     ' latest matrix shape: '+str(matrix.shape))
        if count % 10 == 0:
            print(str(count*100/len(x))+' % готово.')
    print('All trixified!')
    return newlist

def vectorslabels(x): #saves labels and features into separate files for ML usage

    pickle.dump(trixify(preprocformatrices(x)), open("TestFeatures.pkl", "wb")) #saving features into file

    text = open("TestLabels.txt","w") #saving labels into file
    for i in extract_labels(x):
        text.write(i+('\n'))
    print('All done!')

def open_pickle(x): # Loading features file
    with open(x, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
        return file


vectorslabels('Test.txt')

'''
text = open("TrainingLabels.txt","w")

for i in extract_labels('Training.txt'):
    text.write(i + '\n')
'''
# print(open_pickle('TestFeatures1.pkl')[0])