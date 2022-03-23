import re
from nltk import PorterStemmer
import numpy as np

def processEmail(email_contents):
    # This function preprocesses the body of an email and
    # returns a list of indices of the words contained in the email. 
    #

    # Load Vocabulary
    vocabList = {}
    with open('vocab.txt', 'r') as vocabFile:
        for line in vocabFile.readlines():
            i, word = line.split()
            vocabList[word] = int(i)

    # Init return value
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.find("\n\n")
    # if hdrstart:
    #     email_contents = email_contents[hdrstart:]

    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


    # ========================== Tokenize Email ===========================

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')

    # Process file
    l = 0

    # Slightly different order from matlab version

    # Split and also get rid of any punctuation
    # regex may need further debugging...
    email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)

    for token in email_contents:

        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)

        # Stem the word 
        token = PorterStemmer().stem(token.strip())

        # Skip the word if it is too short
        if len(token) < 1:
           continue

        # Look up the word in the dictionary and add to word_indices if
        # found
        # ====================== YOUR CODE HERE ======================
        # Instructions: Fill in this function to add the index of str to
        #               word_indices if it is in the vocabulary. At this point
        #               of the code, you have a stemmed word from the email in
        #               the variable str. You should look up str in the
        #               vocabulary list (vocabList). If a match exists, you
        #               should add the index of the word to the word_indices
        #               vector. Concretely, if str = 'action', then you should
        #               look up the vocabulary list to find where in vocabList
        #               'action' appears. For example, if vocabList{18} =
        #               'action', then, you should add 18 to the word_indices 
        #               vector (e.g., word_indices = [word_indices ; 18]; ).
        # 
        # Note: vocabList{idx} returns a the word with index idx in the
        #       vocabulary list.
        # 
        # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
        #       str2). It will return 1 only if the two strings are equivalent.
        #


        # =============================================================


        # Print to screen, ensuring that the output lines are not too long
        if l + len(token) + 1 > 78:
            print("")
            l = 0
        print('{:s}'.format(token)),
        l = l + len(token) + 1

    # Print footer
    print('\n\n=========================\n')

    return word_indices

def emailFeatures(word_indices):
    # This function takes in a word_indices vector and produces a feature vector from the word indices. 

    n = 1899 # Total number of words in the dictionary
    x = np.zeros((n, 1)) # You need to return this correctly.

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return a feature vector for the
    #               given email (word_indices). To help make it easier to 
    #               process the emails, we have have already pre-processed each
    #               email and converted each word in the email into an index in
    #               a fixed dictionary (of 1899 words). The variable
    #               word_indices contains the list of indices of the words
    #               which occur in one email.
    # 
    #               Concretely, if an email has the text:
    #
    #                  The quick brown fox jumped over the lazy dog.
    #
    #               Then, the word_indices vector for this text might look 
    #               like:
    #               
    #                   60  100   33   44   10     53  60  58   5
    #
    #               where, we have mapped each word onto a number, for example:
    #
    #                   the   -- 60
    #                   quick -- 100
    #                   ...
    #
    #              (note: the above numbers are just an example and are not the
    #               actual mappings).
    #
    #              Your task is take one such word_indices vector and construct
    #              a binary feature vector that indicates whether a particular
    #              word occurs in the email. That is, x(i) = 1 when word i
    #              is present in the email. Concretely, if the word 'the' (say,
    #              index 60) appears in the email, then x(60) = 1. The feature
    #              vector should look like:
    #
    #              x = [ 0 0 0 0 1 0 0 0 ... 0 0 0 0 1 ... 0 0 0 1 0 ..];
    #
    #

    # =========================================================================

    return x
