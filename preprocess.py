
import numpy as np
from __future__ import division
import itertools
import os
import time
import sys
import os
import pandas as pd
os.chdir('/home/prudhvi/Documents')

emails = pd.read_csv('enron_emails.csv')

print(emails['message'][0])