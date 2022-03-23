#  Spam Classification with SVMs
#
#  depends on
#     NLTK package (for Porter stemmer)
#

import scipy.io
import numpy as np
import ex6utils
import ex6modules_spam


def readFile(filename):
    try:
        with open(filename, 'r') as openFile:
            file_contents = openFile.read()
    except:
        file_contents = ''
        print('Unable to open {:s}'.format(filename))
    return file_contents


## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email.

print('Preprocessing sample email (emailSample1.txt)')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = ex6modules_spam.processEmail(file_contents)

# Print Stats
print('Word Indices: ')
print(' {}'.format(word_indices))
print('\n\n')

input('Program paused. Press enter to continue.')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('Extracting features from sample email (emailSample1.txt)')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = ex6modules_spam.processEmail(file_contents)
features      = ex6modules_spam.emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: {:d}'.format( len(features) ) )
print('Number of non-zero entries: {:d}'.format( np.sum(features > 0) ) )

input('Program paused. Press enter to continue.')

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
mat = scipy.io.loadmat('spamTrain.mat')
X = mat["X"]
y = mat["y"]

y = y.flatten()

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
model = ex6utils.svmTrain(X, y, C, "linear")

p = model.predict(X)

print('Training Accuracy: {:f}'.format( np.mean((p == y).astype(int)) * 100 ))

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
mat = scipy.io.loadmat('spamTest.mat')
Xtest = mat["Xtest"]
ytest = mat["ytest"]

ytest = ytest.flatten()

print('Evaluating the trained Linear SVM on a test set ...')

p = model.predict(Xtest)

print('Test Accuracy: {:f}'.format( np.mean((p == ytest).astype(int)) * 100 ))


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtain the vocabulary list
w = model.coef_[0]

# from http://stackoverflow.com/a/16486305/583834
# reverse sorting by index
indices = w.argsort()[::-1][:15]

# Store all dictionary words in dictionary vocabList
vocabList = {}
with open('vocab.txt', 'r') as vocabFile:
    for line in vocabFile.readlines():
        i, word = line.split()
        vocabList[word] = int(i)
vocabList = sorted(vocabList.keys())

print('\nTop predictors of spam: \n')
for idx in indices: 
    print( ' {:s} ({:f}) '.format( vocabList[idx], float(w[idx]) ) )
