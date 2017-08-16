import random
import math
import json
import sys

def unknownwords(trainset, unkThresh = 1, percentUNK = 20):

    unicounts = {}
    with open(trainset, 'r') as f:
        lines = f.readlines()

    for sentence in lines:
        if sentence.isspace():
            continue

        words = sentence.split()
        words = [word.lower() for word in words]

        for word in words:
            unicounts.setdefault(word, 0)
            unicounts[word] += 1 # Increment unigram count for the token

    UNKwords = []
    for key in list(unicounts.keys()):
        if unicounts[key] <= unkThresh:
            UNKwords.append(key)

    indices = random.sample(range(0, len(UNKwords)), round(len(UNKwords)*percentUNK/100))
    return [UNKwords[i] for i in indices]

def replacewithUNK(corpus, UNKwords):

    UNKedcorpus = []
    with open(corpus,'r') as f:
        lines = f.readlines() # Reading one sentence at a time

    for sentence in lines:
        words = sentence.split() # Break sentence into tokens (some punctuations are considered as words)

        tokens = []
        for word in words:
            word = word.lower()
            if word in UNKwords:
                tokens.append('UNK')
            else:
                tokens.append(word)

        nline = ' '.join(tokens)
        UNKedcorpus.append(nline)

    return UNKedcorpus

def countNGramFreq(corpus, n=3):

    wordcounts = {}
    vocabulary = []
    for sentence in corpus:
        if sentence.isspace():
            continue

        words = sentence.split()
        words = ['START'] + words
        words.append('STOP')

        if n >= 1:
            for i in range(0, len(words)):
                unigram = words[i]
                wordcounts.setdefault(unigram, 0)
                wordcounts[unigram] += 1

                if unigram not in vocabulary:
                    vocabulary.append(unigram)
        if n >= 2:
            for i in range(0, len(words)-1):
                bigram = ' '.join(words[i:i+2])
                wordcounts.setdefault(bigram, 0)
                wordcounts[bigram] += 1
        if n >= 3:
            for i in range(0, len(words)-2):
                trigram = ' '.join(words[i:i+3])
                wordcounts.setdefault(trigram, 0)
                wordcounts[trigram] += 1

    return (vocabulary, wordcounts)

def logProbAddK(sequence, counts, n, vocabulary, K=0):

    if len(sequence) == 0:
        return 0
    
    words = []
    words.append('START')
    for i in range(0, len(sequence)):
        word = sequence[i].lower()
        if word not in vocabulary:
            words.append('UNK')
        else:
            words.append(word)
    words.append('STOP')

    logprob = 0
    if n == 1:
        denom = sum([counts[word] for word in vocabulary]) + K*len(vocabulary)
        for i in range(0, len(words)):
            try:
                num = counts[words[i]] + K
            except KeyError:
                num = 0
            prob = num/denom
            logprob += math.log(prob, 2)
    else:
        for i in range(0, len(words)-n+1):
            try:
                num = counts[' '.join(words[i:(i+n)])] + K
            except KeyError:
                num = K
            try:
                denom = counts[' '.join(words[i:(i+n-1)])] + K*(len(vocabulary))
            except KeyError:
                denom = K*len(vocabulary)
            prob = num/denom
            logprob += math.log(prob, 2)

    return logprob

def perplexity(corpus, counts, n, vocabulary, K=0):

    with open(corpus, 'r') as f:
        lines = f.readlines()

    logpp = 0
    i = 0
    M = 0
    for sentence in lines:
        i += 1
        words = sentence.split()
        logpp += logProbAddK(words, counts, n, vocabulary, K)
        M += (len(words) + 2)
    logpp = logpp/M
    pp = math.pow(2, -logpp)
    print("Perplexity " + str(n) + "-gram Language Model = " + str(pp))

    return pp

def linearIntNGram(sequence, counts, vocabulary, lambda1, lambda2, n=3):

    if len(sequence) == 0:
        return 0
    
    words = ['START']
    for i in range(0, len(sequence)):
        word = sequence[i].lower()
        if word not in vocabulary:
            words.append('UNK')
        else:
            words.append(word)
    words.append('STOP')

    logprob = 0
    unigramtotal = sum([counts[word] for word in vocabulary])
    prob = 0
    for i in range(-2, len(words)-n+1):
        try:
            count = counts[' '.join(words[i:(i + n)])]
        except KeyError:
            count = 0
        try:
            totalcount = counts[' '.join(words[i:(i + n - 1)])]
        except KeyError:
            totalcount = 0
        if totalcount == 0:
            prob = 0
        else:
            prob = count/totalcount
        prob3 = prob*(1 - lambda1 - lambda2)

        try:
            count = counts[' '.join(words[i + 1:(i + n)])]
        except KeyError:
            count = 0
        try:
            totalcount = counts[' '.join(words[i + 1:(i + n - 1)])]
        except KeyError:
            totalcount = 0
        if totalcount == 0:
            prob = 0
        else:
            prob = count/totalcount
        prob2 = prob * lambda2

        try:
            count = counts[words[i + 2]]
        except KeyError:
            count = 0
        prob = count/unigramtotal
        prob1 = prob * lambda1

        total_prob = prob1 + prob2 + prob3
        logprob += math.log(total_prob, 2)

    return logprob


def perplexityLinearIntp(corpus, counts, vocabulary, lambda1, lambda2, n=3):

    with open(corpus, 'r') as f:
        lines = f.readlines()

    logpp = 0
    i = 0
    M = 0
    for sentence in lines:
        # print(i)
        i += 1
        words = sentence.split()
        logpp += linearIntNGram(words, counts, vocabulary, lambda1, lambda2, n)
        M += (len(words) + 2) # 'START' and 'STOP'
    logpp = logpp/M
    pp = math.pow(2, -logpp)
    print("Perplexity with lambda1 = " + str(lambda1) + " lambda2 = " + str(lambda2) + " lambda3 = " + str(1 - lambda1 - lambda2) + " is = " + str(pp))
    return pp

def linearIntp(corpus, counts, vocabulary, n=3):

    for i in [0.3]:
        for j in [0.6]:

            if i+j > 1:
                continue
            else:
                pp = perplexityLinearIntp(corpus, counts, vocabulary, i, j, n)

    return pp

if __name__ == "__main__":

    phase = sys.argv[1]

    traincorpus = "brown.train.txt"
    testcorpus = "brown.test.txt"
    devcorpus = "brown.dev.txt"

    if phase == 'prepare':
        unklist = unknownwords(traincorpus, 2, 15)
        new_corpus = replacewithUNK(traincorpus, unklist)
        vocabulary, nGramCounts = countNGramFreq(new_corpus, 3)

        with open('ngram.json', 'w') as lm:
            json.dump(nGramCounts, lm)
        with open('vocablist.txt', 'w') as v:
            v.write(' '.join(vocabulary))
        with open('unklist.txt', 'w') as u:
            u.write(' '.join(unklist))

    elif phase == 'train':

        with open('ngram.json','r') as lmfile:
            nGramCounts = json.load(lmfile)
        with open('vocablist.txt', 'r') as voc:
            vocabulary = voc.read().split()
        print(len(vocabulary))

        print("N-Gram Perplexities without Smoothing")
        logpp = perplexity(traincorpus, nGramCounts, 1, vocabulary, 0)
        logpp = perplexity(traincorpus, nGramCounts, 2, vocabulary, 0)
        logpp = perplexity(traincorpus, nGramCounts, 3, vocabulary, 0)
        print("\nTrigram Perplexities with Add-K Smoothing K = 1")
        logpp = perplexity(traincorpus, nGramCounts, 3, vocabulary, 1)
        print("\nTrigram Perplexities with Add-K Smoothing K = 0.01")
        logpp = perplexity(traincorpus, nGramCounts, 3, vocabulary, 0.01)
        print("\nTrigram Perplexities with Add-K Smoothing K = 0.001")
        logpp = perplexity(traincorpus, nGramCounts, 3, vocabulary, 0.001)
        print("\nLinear interpolation")
        linearIntp(traincorpus, nGramCounts, vocabulary)

    elif phase == 'test':

        with open('ngram.json','r') as lmfile:
            nGramCounts = json.load(lmfile)
        with open('vocablist.txt', 'r') as voc:
            vocabulary = voc.read().split()

        print("Unigram Perplexity without Smoothing")
        logpp = perplexity(testcorpus, nGramCounts, 1, vocabulary, 0)
        print("\nTrigram Perplexities with Add-K Smoothing K = 0.001")
        logpp = perplexity(testcorpus, nGramCounts, 3, vocabulary, 0.001)
        print("\nLinear interpolation")
        linearIntp(testcorpus, nGramCounts, vocabulary)

    elif phase == 'dev':

        with open('ngram.json','r') as lmfile:
            nGramCounts = json.load(lmfile)
        with open('vocablist.txt', 'r') as voc:
            vocabulary = voc.read().split()

        print("Unigram Perplexity without Smoothing")
        logpp = perplexity(devcorpus, nGramCounts, 1, vocabulary, 0)
        print("\nTrigram Perplexities with Add-K Smoothing K = 0.001")
        logpp = perplexity(devcorpus, nGramCounts, 3, vocabulary, 0.001)
        print("\nLinear interpolation")
        linearIntp(devcorpus, nGramCounts, vocabulary)

    else:
        print("Invalid input")
