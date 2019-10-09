"""la idea principal es tomar una combinacion de features y ver como se comparan en visualizacion con los del paper
    luego hacer clasificacion para saber como se comportan, y luego se hace una seleccion de best fatures como en el paper

    domain token length, tld, number of dots in url, symbol count domain, ratios between parts of url (del paper de canada url)
    length ratio url/path, susp words, euclidean distance, kolmogorov smirnov, lullback liebler, (del paper de fabio)
    is IP, network protocol (http,https), length of each part of URL, (huaping yuan)
     """
from scipy import stats
import numpy as np
import pandas as pd

url = "http://www.thenathanbaker.com/thenathanbaker.com/wordpress/wp-content/themes/graphene/style.css?ver=4.0.5"


def partitionURL(url):
    url = str(url)
    protocol = url.split("//")
    print(protocol)

partitionURL(url)


def characterFrequency(url):#to count the frequency of each letter in the given URL
    abecedary = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,
                'm':0,'n':0,'o':0,'p':0,'q':0,'r':0,'s':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0}
    flag = True
    for character in url:
        character = str(character).lower()
        if character in abecedary:
            abecedary[character] += 1
            flag = False #to know if there is a url without letters
    if flag:
        abecedary['a'] = 1
    valuesArray = []
    totalFreq = 1
    for letter in abecedary:
        totalFreq = totalFreq + abecedary[letter]
    for letter in abecedary:
        if totalFreq > 0:
            abecedary[letter] = (abecedary[letter] * 100)/totalFreq
            valuesArray.append(abecedary[letter])
    valuesArray = np.asarray(valuesArray)
    #valuesArray = np.sort(valuesArray)
    return valuesArray



def englishFrequency():#to return the frequency of each letter in the english abecedary
    englishFreq = {'a':8.12,'b':1.49,'c':2.71,'d':4.32,'e':12.02,'f':2.3,'g':2.03,'h':5.92,'i':7.31,
                    'j':0.1,'k':0.69,'l':3.98,
                'm':2.61,'n':6.95,'o':7.68,'p':1.82,'q':0.11,'r':6.02,
                's':6.28,'t':9.1,'u':2.88,'v':1.11,'w':2.09,'x':0.17,'y':2.11,'z':0.07}
    valuesEnglish = []
    for letter in englishFreq:
        valuesEnglish.append(englishFreq[letter])
    valuesEnglish = np.asarray(valuesEnglish)
    #valuesEnglish = np.sort(valuesEnglish)
    return valuesEnglish


def kolmogorovEng(url):#to know if a given data comes from a determined distribution
    kolmogorov = "none"
    frequencies = characterFrequency(url)
    #print(frequencies)
    valuesEnglish = englishFrequency()
    ksTest = stats.ks_2samp(frequencies,valuesEnglish)
    kolmogorov = str(ksTest[0])
    return kolmogorov


def kullbackDivergenceEng(url): #determines how far a distribution is from other
    kullback = "none"
    np.seterr(divide='ignore', invalid='ignore')#to avoid error from divided by zero
    frequencies = characterFrequency(url)
    valuesEnglish = englishFrequency()
    klDiv = stats.entropy(frequencies,valuesEnglish)
    kullback = str(klDiv)
    return kullback

