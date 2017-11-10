
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

os.chdir('/home/prudhvi/Documents')

emails = pd.read_csv('enron_emails.csv')

print(emails['message'][0])

emails.head()

## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
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




