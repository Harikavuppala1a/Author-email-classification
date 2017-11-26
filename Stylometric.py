
import numpy as np
from __future__ import division
import itertools
import os
import time
import sys
import os
import pandas as pd
import  email
import re
from nltk import tokenize
import string
from collections import Counter
from nltk.corpus import stopwords
stops = stopwords.words('english')
import nltk
from nltk import word_tokenize
import operator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
os.chdir('/home/prudhvi/Documents')

emails = pd.read_csv('enron_emails.csv')

print(emails['message'][0])

emails.head()

## Helper functions

'''To get the content from email objects'''

def get_text_from_email(msg):

    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )

    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

# Parse the emails into a list email objects
messages = list(map(email.message_from_string, emails['message']))
emails.drop('message', axis=1, inplace=True)
# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    emails[key] = [doc[key] for doc in messages]
# Parse content from emails
emails['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
emails['From'] = emails['From'].map(split_email_addresses)
emails['To'] = emails['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
emails['user'] = emails['file'].map(lambda x:x.split('/')[0])
del messages


emails['email_length'] = emails.apply(lambda x : len(x['content']) , axis = 1)

emails['digits_count'] = emails.apply(lambda x : sum(c.isdigit() for c in x['content']) , axis = 1)

emails['spaces_count'] = emails.apply(lambda x : sum(c.isspace() for c in x['content']) , axis = 1)

emails['word_count'] = emails.apply(lambda x : len(re.findall(r'\w+', x['content'])) , axis = 1)


def avg_word_length(row):
    filtered_sentence = ''.join(filter(lambda x: x not in '".,;!-', row['content']))
    words = filtered_sentence.split()
    try :
        avg = sum(map(len, words)) / len(words)
    except :
        avg = 0

    return avg

emails['avg_word_length'] = emails.apply(lambda x : avg_word_length(x) ,axis = 1)

def avg_sent_length(row) :
    sents = tokenize.sent_tokenize(row['content'])
    lengths_of_sents = [len(k) for k in sents]
    avg_length = sum(lengths_of_sents) / len(sents)
    return  avg_length


emails['avg_sentence_length'] = emails.apply(lambda x : avg_sent_length(x) ,axis = 1)


# normalising all with length of email

emails['digits_count']  = emails['digits_count'] / emails['email_length']


emails['spaces_count']  = emails['spaces_count'] /emails['email_length']

emails['word_count'] = emails['word_count'] /emails['email_length']

special_chars = string.punctuation


emails['spl_char_count'] = emails.apply(lambda  x : sum(v for k, v in Counter(x['content']).items() if k in special_chars) , axis = 1)

emails['spl_char_count'] = emails['spl_char_count'] / emails['email_length']

emails['paras_count'] = emails.apply(lambda x : x['content'].count('\n\n') + 1 ,axis = 1)

emails['avg_sentences_in_para'] = emails.apply(lambda x :len(re.findall(r'\.',x['content']))/x['paras_count'],axis = 1)




def ratio_fun_words(row) :
    count = len([i for i in word_tokenize(row['content'].lower()) if i in stops])
    try :
        count = count/(row['word_count']*row['email_length'])
    except :
        count = 0

    return  count



emails['ratio_of_fun_words'] = emails.apply(lambda x : ratio_fun_words(x),axis = 1)

list_of_test_authors = ['beck-s','farmer-d','kaminski-v','kitchen-l','lokay-m','sanders-r','williams-w3' ]


emails_sample = emails[emails['user'].isin(list_of_test_authors)]


kk = dict(Counter(emails_sample['X-From']))

sorted_x = sorted(kk.iteritems(), key=operator.itemgetter(1), reverse= True)

list_of_X_from = ['Sally Beck','Richard B Sanders','Daren J Farmer','Kaminski, Vince J </O=ENRON/OU=NA/CN=RECIPIENTS/CN=VKAMINS>','Kitchen, Louise </O=ENRON/OU=NA/CN=RECIPIENTS/CN=LKITCHEN>','Lokay, Michelle </O=ENRON/OU=NA/CN=RECIPIENTS/CN=MLOKAY>','Williams III, Bill </O=ENRON/OU=NA/CN=RECIPIENTS/CN=BWILLIA5>']

emails_sample = emails_sample[emails_sample['X-From'].isin(list_of_X_from)]


final_df = emails_sample.iloc[:,[18,19,20,21,22,23,24,25,26,27]]

target = emails_sample.iloc[:,9]

le = preprocessing.LabelEncoder()

labelled_target = le.fit_transform(target)



X_train,X_test,Y_train,Y_test = train_test_split(final_df,target,test_size= 0.33)

rf = RandomForestClassifier( random_state=0)

parameters = [{ 'n_estimators' : [10,30,50], 'min_samples_split' : [0.1,0.3] , 'min_samples_leaf' : [5,15,20] , 'max_features': [3,5,7] , 'max_depth' :[10,15,30]}]

clf = GridSearchCV(rf, param_grid= parameters, cv=5,
                   scoring= 'accuracy')

clf.fit(X_train,Y_train)

print clf.best_params_


rf = RandomForestClassifier(max_features= 8, random_state=0)

rf.fit(X_train, Y_train)

preds = rf.predict(X_test)



accuracy_score(Y_test,preds)
print classification_report(Y_test,preds)

cm = confusion_matrix(Y_test,preds)
print cm

for i in range(len(cm)):
    val = float(cm[i][i]) / float((cm[i][0] + cm[i][1] + cm[i][2] + cm[i][3] + cm[i][4] + cm[i][5] + cm[i][6]))
    print val
