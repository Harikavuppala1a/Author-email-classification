# Author-Email-Classification

Create the folder Databse and place seven author folders ( b-s, f-d, k-v, k-l, l-m, s-r, w-w3) in it from enron databse and run the programs.

Stylometric.py:

Run: Python stylometric.py
 After running this file we get a feature vector for each email which contains information about sylometric features.
 The features extracted here are: Email length in characters, Digit density, Ratio of space to email length, Each special character count, Document length in words, Average length of words, Average sentence length, Number of paragraphs in documents, Number of sentences in paragraphs,Frequencies of function words. 
 
There are two folders here: sub+body and sub+body+participants
Both the folders have all these files. sub+body is we are extracting the context specific features only from the subject and body part of an email. In sub+body+participants we are extracting the context specific features from the subject, body, participants (cc,bcc,to) from an email.
                 
– twf.py : The term weight filtering (TWF), which is a simple method in which features are ranked according to their weight, which is usually the TF of TF-IDF.

– df.py : The document frequency (DF) method based on the Information Retrieval scoring scheme, ranks features according to the number of different documents where the feature occurs.

– var.py : Variance, which can also be used for determining the distance of a feature to the collection’s mean. It is given by V (x) = E{(x − μ x )2}, where V(x) is the variance of the feature x, E is the expectation operator and μ x is the mean of feature x.

– ig.py : The information gain (IG) measures the number of bits of information obtained for classification by knowing the presence or absence of a feature. The information gain of a feature/term t k is given by IG(t_k ) = H(C) − H(C|t_k ), where H(C) is the entropy of class C and H(C|t_k ) is the conditional entropy of the class, given term t_k .

– l0norm.py : The L0 -norm works by computing the rank of the remaining features. 

- fisher.py : The Fisher criterion (FC) is used in Fisher’s linear discriminant and applied to FS as a method that tries to evaluate how much a feature can discriminate between two classes. The result of the computation is a matrix of pairs of class-feature. An analysis on this matrix must be done in order to determine the scalar value of the feature. 
 
 
 

