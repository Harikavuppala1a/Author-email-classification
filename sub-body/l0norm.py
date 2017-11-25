#!/usr/local/bin/python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import shutil
import string
import fnmatch
import os, errno
import re
import sys
from collections import Counter
from email.parser import Parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk import PorterStemmer

PS = PorterStemmer()
punctuation='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
removepunctuation = dict((ord(char), 32) for char in punctuation)
datadir = 'database/'
raw_path = ['kitchen-l/','farmer-d/','beck-s/','kaminski-v/','sanders-r/','lokay-m/', 'williams-w3/']
dest ='predatabase/'
dest1 = 'maindatabase/'
inputdir =['train/','test/']


def make_featurevector(inputdata, classid,worddictionary,dics):
                temp = []
                N = 75210
                Countwords = 0
                uniword, unifreq = worddictionary.keys(), worddictionary.values()
                for k in range(len(uniword)):
                           wordT = uniword[k].strip()
                           if wordT in inputdata:
                             Countwords = dics[wordT]+1

                           else:
                             Countwords = 0
                           temp.append(Countwords)                  
                   
                 
                temp = map(str,temp) 
                return temp, classid


def testfeat(uni,dics):
  
  X_training = []
  Y_training = []
  X_testing = []
  Y_testing = []
  countertest =Counter() 
  for fol in inputdir:
      classid = -1
      for c in raw_path:
            classid =classid+1
            for dirpath, dirs, files in os.walk("maindatabase/"+fol+c):
                            for filename in fnmatch.filter(files, '*'):
                                      with open(os.path.join(dirpath, filename)) as f:
                                               emaildata =f.read()
                                               email = Parser().parsestr(emaildata)
                                               filtered_subj = preunique(email['subject'])
                                               filtered_body = preunique(email.get_payload())
                                               filtered = filtered_subj + filtered_body
                                               countertest = Counter(filtered)
                                               if fol == 'train/':
                                               #print countertest                  
                                                      X_train,Y_train= make_featurevector(countertest,classid,uni,dics)
                                                      X_training.append(X_train)
                                                      Y_training.append(Y_train)
                                               else:
                                                      X_test,Y_test= make_featurevector(countertest,classid,uni,dics)
                                                      X_testing.append(X_test)
                                                      Y_testing.append(Y_test)

  #print len(Y_testing), len(X_testing)
  #print len(Y_training), len(X_training)
  return X_training, Y_training, X_testing, Y_testing



def preunique(unidata):
         filtereddata1 = []
         stop_words = set(stopwords.words('english'))
         if unidata:

             subj1 = unidata
             shortword = re.compile(r'\W*\b\w{1,3}\b')
             subj1 = shortword.sub('', subj1)
             subj1 = re.sub(r'\w*_\w*', '', subj1)
             subj1 = re.sub(r'\w*[0-9]\w*', '', subj1)
             subj=[e.lower() for e in map(string.strip, re.split("(\W+)", subj1)) if len(e) > 0 and not re.match("\W",e)]
             filtereddata1 = [w for w in subj if not w in stop_words and not w.isdigit()]
            # for tokens in filtereddata:
             #          toens=PS.stem(tokens)
              #         filtereddata1.append(tokens)
                 
         return filtereddata1
def getuni():
 dicti={}
 counter = Counter()
 for dirpath, dirs, files in os.walk("maindatabase/"+inputdir[0]):
    
    for filename in fnmatch.filter(files, '*'):
       
        with open(os.path.join(dirpath, filename)) as f:
            emaildata =f.read()
            email = Parser().parsestr(emaildata)
            filtered_subj = preunique(email['subject'])
            filtered_body = preunique(email.get_payload())
            filtered = filtered_subj + filtered_body
            chj = Counter(filtered)
            for datas in chj:
                if datas in dicti:
                             dicti[datas] = dicti[datas] + 1
                else:
                             dicti[datas] = 1
                                                 
            counter.update(filtered)
 counter = Counter({k: c for k, c in counter.iteritems() if ((c > 1000) & (c<8000))})
           # counter.update(filtered_body)
 
 return counter,dicti

def folderdata():
    ylabel = []
    z = -1
    for c in raw_path:
        z =z +1
        i = 0
        try:
           os.makedirs(dest+c)
        except OSError as e:
           if e.errno != errno.EEXIST:
              raise
        for dirpath, dirs, files in os.walk("database/"+c):

             for filename in fnmatch.filter(files, '*.'):
                  i = i + 1
                  ylabel.append(z)
                  filenames = os.path.join(dirpath, filename)
                  filed = str(i)
                  shutil.copy2(filenames, dest+c+filed)

    return ylabel


def splitdata():
  
 
       for c in raw_path:
            filenames = []
            print c
            i =-1
          
            try:
                 os.makedirs(dest1+inputdir[0]+c)
                 os.makedirs(dest1+inputdir[1]+c)
            except OSError as e:
                 if e.errno != errno.EEXIST:
                        raise
            for dirpath, dirs, files in os.walk("predatabase/"+c):

                    for filename in fnmatch.filter(files, '*'):
                           i = i + 1
                           #with open(os.path.join(dirpath, filename)) as f:
            
                           filenames.append(os.path.join(dirpath, filename))
                           #print filenames

                    for val in range(len(files)):

                           if val < int(len(files)*0.8):    
                                  shutil.copy2(filenames[val], dest1+inputdir[0]+c)
                           else:
                                  shutil.copy2(filenames[val], dest1+inputdir[1]+c)

def calculate_accuracy(predictions, labels):
         return accuracy_score(labels, predictions)


if __name__ == "__main__":


# labl = folderdata()
 #print labl
 #splitdata()
 uni, dicti = getuni()
 #print uni
 print len(uni)
 print len(dicti)
 X_traindt, Y_traindt, X_testdt, Y_testdt = testfeat(uni,dicti)
 print "over"
 X_traindt = np.array(X_traindt).astype(np.float)
 Y_traindt = np.array(Y_traindt).astype(np.float)
 X_testdt = np.array(X_testdt).astype(np.float)
 Y_testdt = np.array(Y_testdt).astype(np.float)
 ####NB
 #gnb = GaussianNB()
 #gnb.fit(X_traindt, Y_traindt)
 #y_pred=gnb.predict(X_testdt)
 
########SVM
# svm
 #clf = svm.SVC()
 #clf.fit(xtrain, ytrain) 
# y_pred=clf.predict(xtest) 
###########LOGISTIC REGRESSION
 reg=LogisticRegression(multi_class='multinomial',solver ='newton-cg')
 reg.fit(X_traindt,Y_traindt)
 y_pred=reg.predict(X_testdt)



 accuracy=calculate_accuracy(y_pred,Y_testdt)
 print accuracy
 cm = confusion_matrix(Y_testdt, y_pred)
 print cm
 for i in range(len(cm)):
   val = float(cm[i][i])/float((cm[i][0]+cm[i][1]+cm[i][2]+cm[i][3]+cm[i][4]+cm[i][5]+cm[i][6]))
   print val
