import re
import gensim
import numpy as np
import pymorphy2
import string
import pickle
from pymystem3 import Mystem

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
    return q

def extract_labels(x):
    lines = [line.rstrip('\n') for line in open(x, encoding='utf-8-sig')]
    q = []
    for i in lines:
        i = i.split('\t')
        q.append(i[1])
    return q

def preproc(x):
    list = []

    for i in x:
        i = i.rstrip().lower().replace('кое-','кое- ').replace('-то',' -то').replace('-либо',' -либо').\
            replace('-нибудь',' -нибудь').replace('-таки',' -таки').replace('ё','е')
        i = ''.join(ch for ch in i if ch not in set(string.punctuation)).replace('»', '').replace('«', '').replace('—', '').replace('–', '')
        i = ' '.join(i.split())
        list.append(i)

    return list

def POS_tagger(i): #for word2vec single sentence processing

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

    if x in model:
        vector = model[x]
    else:
        vector = np.zeros((300,))

    return vector

def determinefeatures(a): #extracting regexp features from single word

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

    return list

def open_pickle(x):
    with open(x, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
        return file


def wordpipeline(x, d):
    filename = x + '.txt'
    labels = extract_labels(filename)
    labelsname = x + 'Labels.txt'
    savelabs = open(labelsname, "w")  # saving labels into file
    for i in labels:
        savelabs.write(i + ('\n'))
    print('Labels saved!\n---------------------------\n')

    questions = extract_questions(filename)

    rawquestions = preproc(questions)
    rawdataname = x+'Raw.pkl'
    pickle.dump(rawquestions, open(rawdataname, 'wb')) # preprocessed questions for next step
    print('Raw dumped!\n---------------------------\n')

    pos = []
    counter = 1
    for i in rawquestions:
        pos.append(POS_tagger(i)) #POS tagging
        if counter%10 == 0:
            print(counter)
        counter += 1

    posdataname = x+'POS.pkl'
    pickle.dump(pos, open(posdataname, 'wb'))
    print('POS dumped!\n---------------------------\n')

    for i in range(len(pos)):
        parts = pos[i]
        words = rawquestions[i].split(' ')
        for m in range(len(parts)):
            emb = np.append(vectorize(parts[m]), determinefeatures(words[m])) # creating word vectors
            if pos[i][m] not in d: # adding vectors to dictionary
                d[pos[i][m]] = emb

    pickle.dump(d, open('FeaturesDictionary.pkl', 'wb'))
    print('Dictionary created!\n---------------------------\n')
    print('All done!')

embs = open_pickle('FeaturesDictionary.pkl')

wordpipeline('Training', embs)

