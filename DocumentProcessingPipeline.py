import re
import fasttext
import gensim
from gensim.models import doc2vec, utils
import numpy as np
import pymorphy2
import string
from pymystem3 import Mystem

def extract_questions(x): #x - labeled data name, here separator is \t
    lines = [line.rstrip('\n') for line in open(x)]
    q = []
    for i in lines:
        i = i.split('\t')
        q.append(i[0])
    return q

def extract_labels(x):
    lines = [line.rstrip('\n') for line in open(x)]
    q = []
    for i in lines:
        i = i.split('\t')
        q.append(i[1])
    return q

def POS_tagger(x): #for word2vec model processing
    morph = pymorphy2.MorphAnalyzer()
    m = Mystem()
    translator = str.maketrans('', '', string.punctuation)

    text1 = []
    for i in x:
        i = i.rstrip()
        i = i.translate(translator)
        i = i.replace('  ', '')
        text1.append((m.lemmatize(i))) #sentence to list of lemmas

    text2 = []
    for i in text1:
        text2.append(i[:-1])
    for i in text2:
        for a in i:
            if a == ' ':
                i.remove(a)

    text3 = []
    for i in text2:
        sent = []
        for a in i:
            a = a + '_' + str((morph.parse(a)[0]).tag.POS)
            sent.append(a)
        text3.append(sent) #POS tagging
    return text3

def vectorize(x): #presenting sentence as vector
    model = gensim.models.KeyedVectors.load_word2vec_format('ruscorpora_upos_skipgram_300_10_2017.bin.gz',
                                                            binary=True) # pre-trained vector model for extracting features

    sent1 = []
    for s in x:
        sent2 = []
        for i in s:
            if i in model:
                vector = model[i]
                sent2.append(vector)
            else:
                continue
        sent1.append(sent2) #vectors from words in list

    sent3 = []
    for i in sent1:
        if len(i) == 0: #if no words have been found in pre-trained model
            i = np.zeros((300,))
            sent3.append(i)
        else:
            sent3.append(np.mean(i, axis=0)) #one-vector sentence representation
    return sent3

def determinefeatures(a): #extracting regexp features from single question
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
    list = []

    m = re.match(r'.*[\?\.]', a)
    if m:
        t1 = 1

    m = re.match(r'.* ((ли)|(не)) .*[\?\.]', a)
    if m:
        t2 = 1

    m = re.match(r'.*[кК]то.*[\?\.]', a)
    if m:
        t7 = 1

    m = re.match(r'.*[чЧ]то.*[\?\.]', a)
    if m:
        t7 = 1

    m = re.match(r'.*[гГ]де.*[\?\.]', a)
    if m:
        t7 = 1

    m = re.match(r'.* ([иИ]ли) .*[\?\.]', a)
    if m:
        t6 = 1

    m = re.match(r'.*(([пП]очему)|([чЧ]то было|явилось|стало|(по?)служило причиной)).*[\?\.]', a)
    if m:
        t13 = 1

    m = re.match(r'.*([чЧ]то означает|такое|значит).*[\?\.]', a)
    if m:
        t3 = 1

    m = re.match(r'.*([сС]колько|[кК]ак много).*[\?\.]', a)
    if m:
        t9 = 1

    m = re.match(r'.*(((([вВ]о)|([чЧ]ерез)) сколько)|([вВ] (какое|который) (время|час))).*[\?\.]', a)
    if m:
        t7 = 1

    m = re.match(r'.*((([кК]огда)|([кК]акого числа))|([вВ] како[мй] году|месяце|(день недели))).*[\?\.]', a)
    if m:
        t7 = 1

    m = re.match(r'.*(([кК]аковы последствия)|([чЧ]то следует (из|за))|([чЧ]то будет (за|после|дальше|далее))).*[\?\.]', a)
    if m:
        t14 = 1

    m = re.match(r'.*(([кК]ак(ой|ая|ое|ом|ие))|([кК]ак(им|ими|ие) ((свойств(ом|ами|а))|(качеств(ом|ами|а))) (наделен|обладает|характеризуется|отличается|имеет))).*[\?\.]', a)
    if m:
        t8 = 1

    m = re.match(r'.*(([сС] помощью|[пП]осредством)? (([кК]ого)|([чЧ]его))|(([чЧ]ем|[кК]ем) (воспользова(лся|лась|ться))?)|([бБ]лагодаря (кому|чему))|(([кК]то|[чЧ]то) позволил(о?))).*[\?\.]', a)
    if m:
        t11 = 1

    m = re.match(r'.*(([кК] чему)|([зЗ]ачем)|([сС] как(ой|ими) (цел(ью|ями))|(задач(ей|ами)))|([дД]ля чего)|(([вВ]о имя)|([дД]ля как(ой|их)) (цел(и|ей))|(задач(и?)))|([кК]ак(ую|ие) (цел(ь|и))|(задач(у|и)))).*[\?\.]', a)
    if m:
        t12 = 1

    m = re.match(r'.*(([пП]риведи|[кК]акой пример|образец)|(([чЧ]то|[кК]то) ((может ((служить)|(выступ(ать|ить))))|((по?)служит|выступ(ает|ит))) (как )?((пример(ом?))|(образ(ец|цом))))).*[\?\.]', a)
    if m:
        t4 = 1

    m = re.match(r'.*(([чЧ]то (надо?).* чтобы|делать)|([кК]ак)|([пП]о как(ому|им) (план(у|ам))|(алгоритм(у|ам)))|([пП]о как(ому|ой) (схем(е|ам))|(тактик(е|ам)))|([кК]ак(им|ими) (образ(ом|ами))|(пут(ем|ями)))|([кК]ак(ой|ими) (тактик(ой|ами))|(схем(ой|ами)))) .*[\?\.]', a)
    if m:
        t10 = 1

    m = re.match(r'.*(([чЧ]ем\S.* похо(ж|жи|жа|же)|отлича(ются|ется|))|[кК]ак\S.* похо(ж|жи|жа|же)| отлича(ются|ется|)).*[\?\.]', a)
    if m:
        t5 = 1

    list.append(t1)
    list.append(t2)
    list.append(t3)
    list.append(t4)
    list.append(t5)
    list.append(t6)
    list.append(t7)
    list.append(t8)
    list.append(t9)
    list.append(t10)
    list.append(t11)
    list.append(t12)
    list.append(t13)
    list.append(t14)
    return list

def extract_features(x): #extracting regexp features from list of questions
    flist = []
    for i in extract_questions(x):
        flist.append(determinefeatures(i))
    return flist

def vecsnfeatures(x): #adding regexp features as a binary sequence to the end of the file
    vectors = vectorize(POS_tagger(extract_questions(x)))
    features = extract_features(x)
    sent1 = []
    for i in range(len(features)):
        y = np.append(vectors[i], features[i])
        sent1.append(y)
    return sent1

def vectorslabels(x): #saves labels and features into separate files for ML usage

    np.savetxt('SampleFeatures.txt',vecsnfeatures(x)) #saving features into file

    text = open("SampleLabels.txt","w") #saving labels into file
    for i in extract_labels(x):
        text.write(i+('\n'))

vectorslabels('TestPipeline.txt') #executing