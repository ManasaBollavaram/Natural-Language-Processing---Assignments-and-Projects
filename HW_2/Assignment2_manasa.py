import re, random
import os, json
from collections import defaultdict
import math
import numpy as np

def unknownwords(trainset, unkThresh=2, percentUNK = 80):

    unicounts = {}
    with open(trainset,'r',encoding="utf8") as f:
        lines=f.readlines()
    for sentence in lines:
        if sentence.isspace():
            continue
        words=sentence.split()
        words=[word.lower() for word in words]
        for word in words:
            unicounts.setdefault(word,0)
            unicounts[word]+=1
            #print(unicounts[word])
    #print("WordVocabcounoccurencestotal=",len(set(list(unicounts.keys()))))
    #print("totalcounts=",sum(unicounts.values()))
    UNKwords = []
    counts = 0
    s=[]
    for val in unicounts.values():
        if val in s:
            continue
        else:
            s.append(val)
    #print(s)
    donotUNK = ['WEBADRESS','USERNAME','NUMBERS','ALPHANUMERIC','HASHTAG','EMPTY','PUNCTUATION']
    for key in list(unicounts.keys()):
        #print(key)
        if key in donotUNK:
            continue
        if unicounts[key] <= unkThresh:
            UNKwords.append(key)
            counts+=unicounts[key]
            #print(counts)
    #print("Number of words less than 2=",len(UNKwords))
    #print("Number of words less than 2 counts=",counts)

    indices = random.sample(range(0, len(UNKwords)), round(len(UNKwords)*percentUNK/100))
    UNKwords=[UNKwords[i] for i in indices]

    with open('unknownwordslist.txt', 'w',encoding="utf8") as f:
        f.write('\n'.join(UNKwords))


    return 'unknownwordslist.txt'

def replacewithUNK(corpusfile, unklistfile, outfile):
      

    with open(unklistfile, 'r',encoding="utf8") as f:
        unklist = f.readlines()
    wordtagset = []
    with open(corpusfile,'r') as f:
        for line in f:
            wordtagset.append(json.loads(line))
    for line in wordtagset:
        for wordtagpair in line:
            word = wordtagpair[0]
            if word in unklist:
                wordtagpair[0] = 'UNK'

    with open(outfile, 'w') as f:
        json.dump(wordtagset, f)


    return

def countNGramFreq(corpus, n=3):
    wordcounts={}
    #vocabulary=[]
    for each_tweet_tagset in corpus:
        words=['START']
        words.extend(each_tweet_tagset)
        words.append('STOP')
        #print(words)
        if n>=1:
            for i in range(0,len(words)):
                unigram=words[i]
                wordcounts.setdefault(unigram,0)
                wordcounts[unigram]+=1
                #if unigram not in vocabulary:
                    #vocabulary.append(unigram)
        if n>=2:
            for i in range(0,len(words)-1):
                bigram=' '.join(words[i:i+2])
                wordcounts.setdefault(bigram,0)
                wordcounts[bigram]+=1
        if n>=3:
            for i in range(0, len(words)-2):
                trigram=' '.join(words[i:i+3])
                wordcounts.setdefault(trigram,0)
                wordcounts[trigram]+=1
    return wordcounts


def transition_probability(poscounts, eachtag_index, tagevaluating_index, uniquetags):
    lambda1 = 0.30
    lambda2 = 0.70
    total_unicounts=0

    for tag in uniquetags:
        total_unicounts = total_unicounts + poscounts[tag]
    total_unicounts += poscounts['STOP']

    if tagevaluating_index >= len(uniquetags):
        tagevaluating = 'STOP'
    else:
        tagevaluating = uniquetags[tagevaluating_index]
    unicounts = poscounts[tagevaluating]
    unigram_probability = unicounts / total_unicounts

    transition_probability = []
    everytag = []

    for i in eachtag_index:

        if i == -1:
            everytag_object = 'START'
        else:
            everytag_object = uniquetags[i]
        everytag.append(everytag_object)
        bigram = ' '.join([everytag_object, tagevaluating])
        bigramcounts = poscounts[bigram]
        unicounts_everytag = poscounts[everytag_object]

        if unicounts_everytag == 0:
            bigram_probability = 0
        else:
            bigram_probability = bigramcounts/unicounts_everytag

        transition_probability.append(math.log(((lambda1*unigram_probability)+(lambda2*bigram_probability)),2))
    #print("transprob",prob)
    return np.array(transition_probability)

def transition_probability_trigram(poscounts, eachtag_index, tagevaluating_index0, tagevaluating_index, uniquetags):
    lambda1 = 0.2
    lambda2 = 0.4
    lambda3 = 0.4
    total_unicounts=0

    for tag in uniquetags:
        total_unicounts = total_unicounts + poscounts[tag]
    total_unicounts += poscounts['STOP']

    if tagevaluating_index >= len(uniquetags):
        tagevaluating = 'STOP'
    else:
        tagevaluating = uniquetags[tagevaluating_index]
    unicounts = poscounts[tagevaluating]
    unigram_probability = unicounts / total_unicounts

    tagevaluating1 = uniquetags[tagevaluating_index0]
    bigram = ' '.join([tagevaluating1, tagevaluating])
    bigramcounts = poscounts[bigram]
    unicounts_tag1 = poscounts[tagevaluating1]

    if unicounts_tag1 == 0:
        bigram_probability = 0
    else:
        bigram_probability = bigramcounts / unicounts_tag1

    everytag = []

    transition_probability = []
    for i in eachtag_index:

        if i == -1:
            everytag_object = 'START'
        else:
            everytag_object = uniquetags[i]

        everytag.append(everytag_object)
        trigram = ' '.join([everytag_object, tagevaluating1, tagevaluating])
        trigramcounts = poscounts[trigram]
        bicounts_tag1 = poscounts[everytag_object + ' ' + tagevaluating1]

        if bicounts_tag1 == 0:
            trigram_probability = 0
        else:
            trigram_probability = trigramcounts/bicounts_tag1

        transition_probability.append(math.log(((lambda1*unigram_probability)+(lambda2*bigram_probability)+(lambda3*trigram_probability)),2))
   
    return np.array(transition_probability)

def emission_probability(wordtagcounts, poscounts, uniquewords, word, tag):

    key = ','.join([word,tag])
    if key in list(wordtagcounts.keys()):
        num = wordtagcounts[key]
    else:
        num = 0
        #print("went",key)
        
    denom = poscounts[tag]

    k = 0.001
    probability = (num+k)/(denom+(k*len(uniquewords)))

    
    return math.log(probability,2)


def viterbi_bigramHMM(onetweet):
    #print('bi')
    with open('POStag_uni_bi_trigram_counts.json', 'r') as f:
        blah = json.load(f)
    POScounts = defaultdict(int, blah)
    with open('uniquetags.txt','r') as f:
        uniquetags = f.read()
    uniquetags = uniquetags.split()
    #print(uniquetags)
    with open('uniquewords.txt', 'r',encoding="utf8") as f:
        uniquewords = f.read()
    uniquewords = uniquewords.split()
    with open('wordtag_model.json', 'r') as f:
        blah = json.load(f)
    WordTagcounts = defaultdict(int, blah)
    Number_of_tags = len(uniquetags)
    Number_of_words_in_tweet = len(onetweet)
    back_pointers = np.ndarray(shape=(Number_of_tags,Number_of_words_in_tweet), dtype=int)
    previous_path_probability = np.zeros(Number_of_tags)
    probability_of_tag = np.zeros(Number_of_tags)
    transition_probability_from_previous_tag = np.zeros(Number_of_tags)


    for i in range(0, Number_of_tags):
        back_pointers[i][0] = -1
        e=emission_probability(WordTagcounts, POScounts, uniquewords, onetweet[0], uniquetags[i])
        previous_path_probability[i] = transition_probability(POScounts, [-1], i, uniquetags)+ e

    
    for i in range(1, Number_of_words_in_tweet):
        for j in range(0, Number_of_tags):
            e = emission_probability(WordTagcounts, POScounts, uniquewords, onetweet[i], uniquetags[j])
            k = np.arange(0,Number_of_tags,1)
            transition_probability_from_previous_tag = transition_probability(POScounts, k, j, uniquetags)
            total_probability = np.add(transition_probability_from_previous_tag, previous_path_probability) + e
            #total_probability=np.absolute(total_probability)
            probability_of_tag[j] = np.amax(total_probability)
            back_pointers[j][i] = np.argmax(total_probability)
            
        previous_path_probability = np.copy(probability_of_tag)
        #print(back_pointers[j][i])

    for i in range(0, Number_of_tags):
        transition_probability_from_previous_tag[i] = transition_probability(POScounts, [i], Number_of_tags, uniquetags)
    total_probability = np.add(transition_probability_from_previous_tag,previous_path_probability)
    back_pointer_end = np.argmax(total_probability)
    #print("term prob",total_probability)

    
    reverse_predicted_tag_sequence = [uniquetags[back_pointer_end]]
    #print(back_pointer_end)
    #print(reverse_predicted_tag_sequence)

    if Number_of_words_in_tweet == 1:
        return reverse_predicted_tag_sequence

    back_pointer_previous_index = np.arange(Number_of_words_in_tweet-1,1,-1)
    for i in back_pointer_previous_index:
        #print(back_pointer_end)
        #print(reverse_predicted_tag_sequence)
        back_pointer_end = back_pointers[back_pointer_end][i]
        reverse_predicted_tag_sequence.append(uniquetags[back_pointer_end])

    if Number_of_words_in_tweet == 2:
        #print("went")
        back_pointer_end = back_pointers[back_pointer_end][1]
    else:
        #print("wenthere")
        back_pointer_end = back_pointers[back_pointer_end][i-1]
    reverse_predicted_tag_sequence.append(uniquetags[back_pointer_end])
    #reverse_predicted_tag_sequence.append('P')
    predicted_tag_sequence=reverse_predicted_tag_sequence[::-1]
    #predicted_tag_sequence=reverse_predicted_tag_sequence.reverse()
    #print(reverse_predicted_tag_sequence)
    #print("now")
    #print(predicted_tag_sequence)
    #print("now1")
    return predicted_tag_sequence

def viterbi_trigramHMM(onetweet):
    with open('POStag_uni_bi_trigram_counts.json', 'r') as f:
        blah = json.load(f)
    POScounts = defaultdict(int, blah)
    with open('uniquetags.txt','r') as f:
        uniquetags = f.read()
    uniquetags = uniquetags.split()
    #print(uniquetags)
    with open('uniquewords.txt', 'r',encoding="utf8") as f:
        uniquewords = f.read()
    uniquewords = uniquewords.split()
    with open('wordtag_model.json', 'r') as f:
        blah = json.load(f)
    WordTagcounts = defaultdict(int, blah)
    Number_of_tags = len(uniquetags)
    Number_of_words_in_tweet = len(onetweet)
    back_pointers = np.ndarray(shape=(Number_of_tags**2,Number_of_words_in_tweet), dtype=int)
    previous_path_probability = np.zeros(Number_of_tags**2)
    probability_of_tag = np.zeros(Number_of_tags**2)
    transition_probability_from_previous_tag = np.zeros(Number_of_tags**2)
    

    
    for i in range(0, Number_of_tags): 
        back_pointers[(Number_of_tags*i):(Number_of_tags*(i+1)),0:1] = np.ones(Number_of_tags).reshape(Number_of_tags,1)*(-1)
        e=emission_probability(WordTagcounts, POScounts, uniquewords, onetweet[0], uniquetags[i])
        previous_path_probability[(Number_of_tags*i):(Number_of_tags*(i+1))] = np.ones(Number_of_tags)*(transition_probability(POScounts, [-1], i, uniquetags)+e)
    

    if Number_of_words_in_tweet>1:
        for i in range(0,Number_of_tags):
            back_pointers[(Number_of_tags*i):(Number_of_tags*(i+1)), 1:2] = np.ones(Number_of_tags).reshape(Number_of_tags, 1) * (-1)
            e = emission_probability(WordTagcounts, POScounts, uniquewords, onetweet[1], uniquetags[i])
            for j in range(0,Number_of_tags):
                trans_prob = transition_probability_trigram(POScounts, [-1], j, i, uniquetags)[0]
                transition_probability_from_previous_tag[(Number_of_tags*i)+j] = trans_prob + previous_path_probability[Number_of_tags*j] + e
        previous_path_probability = np.copy(transition_probability_from_previous_tag)
        
    for i in range(2, Number_of_words_in_tweet):
        for j in range(0, Number_of_tags):
            e = emission_probability(WordTagcounts, POScounts, uniquewords, onetweet[i], uniquetags[j])
            for m in range(0, Number_of_tags):
                k = np.arange(0,Number_of_tags,1)
                transition_probability_from_previous_tag = transition_probability_trigram(POScounts, k, m, j, uniquetags)
                total_probability = np.add(transition_probability_from_previous_tag, previous_path_probability[(Number_of_tags*m):(Number_of_tags*(m+1))]) + e
                probability_of_tag[(Number_of_tags*j)+m] = np.amax(total_probability)
                back_pointers[(Number_of_tags*j)+m][i] = np.argmax(total_probability)
        previous_path_probability = np.copy(probability_of_tag)

    transition_probability_from_previous_tag = np.zeros(Number_of_tags ** 2)
    for i in range(0, Number_of_tags):
        if Number_of_words_in_tweet > 1:
            k = np.arange(0, Number_of_tags, 1)
        else:
            k = np.ones(Number_of_tags)*(-1)

        transition_probability_from_previous_tag[(Number_of_tags*i):(Number_of_tags*(i+1))] = transition_probability_trigram(POScounts, k, i, Number_of_tags, uniquetags)
    total_probability = np.add(transition_probability_from_previous_tag,previous_path_probability)
    backpointer_end = np.argmax(total_probability)


    backpointer_end0 = int(math.floor(backpointer_end/Number_of_tags))
    backpointer_end1 = backpointer_end%Number_of_tags

    if Number_of_words_in_tweet == 1:
        Predicted_tag_sequence = [uniquetags[backpointer_end0]]
        return Predicted_tag_sequence

    Predicted_tag_sequence = [uniquetags[backpointer_end0], uniquetags[backpointer_end1]]

    if Number_of_words_in_tweet == 2:
        return Predicted_tag_sequence

    backpointer_previous_indices = np.arange(Number_of_words_in_tweet-1,1,-1)
    for i in backpointer_previous_indices:
        index = back_pointers[backpointer_end][i]
        Predicted_tag_sequence.append(uniquetags[index])
        backpointer_end = index + Number_of_tags*(backpointer_end%Number_of_tags)
    Predicted_tag_sequence=Predicted_tag_sequence[::-1]
    return Predicted_tag_sequence
def cleantheWord(word):
    if re.match(r'^[A-Za-z0-9]+$', word):
        if re.match(r"^[^0-9]+$", word):
            pass
        else:
            word = 'ALPHANUMERIC'
            return word
    word = word.lower()
    word = re.sub('((www\.[^\s]+)|(https?:\/\/[^\s]+))', 'WEBADRESS', word)
    word = re.sub('(@[^\s]+)|(@[\s][^\s]+)', 'USERNAME', word)
    word = re.sub('[\s]+', ' ', word)
    word = re.sub(r"#[^\s]+", 'HASHTAG', word)
    word =  re.sub("^[\!.;'?()]+$", 'PUNCTUATION', word)
    if re.match(r'^[^A-Za-z]+$', word):
        if re.match(r'[0-9]',word):
            word = 'NUMBERS'
            return word
    if word == '':
        word = 'EMPTY'

    return word

def cleanandUNKTrainset(corpusfile):

    wordtagset = []
    with open(corpusfile,'r') as f:
        for line in f:
            jline = json.loads(line)
            wordtagset.append(jline)
    cleanedlines = []
    
    for lines in wordtagset:
        cleanedlines = []
        for wordtagpair in lines:
            #print(wordtagpair[0])
            word=wordtagpair[0]
            cleanedword=cleantheWord(word)
            cleanedlines.append(cleanedword)
                #print(cleanedlines)
        for i in range(0,len(lines)):
            #print("once")
            lines[i][0] = cleanedlines[i]


    with open('train.clean.json','w') as f:
        json.dump(wordtagset, f)
    words = []
    with open('train.clean.json', 'r') as f:
        for line in f:
            wordtagsentence = json.loads(line)
            #print(line)
            for wordtagpairs in wordtagsentence:
                for wordtag in wordtagpairs:
                    words.append(wordtag[0])
                    #print(xy2[0])


    with open('wordsequence.txt', 'w',encoding="utf8") as f:
        f.write(' '.join(words))

    unknownwordsfile = unknownwords('wordsequence.txt', unkThresh=1, percentUNK = 80)
    with open(unknownwordsfile, 'r',encoding="utf8") as f:
        unknownwordslist = f.readlines()
    
    for line in wordtagset:
        for wordtagpair in line:
            word = wordtagpair[0]
            if word in unknownwordslist:
                wordtagpair[0] = 'UNK'

    with open('train.unk.json', 'w') as f:
        json.dump(wordtagset, f)

    return

def cleanandUNK_dev_or_testset(corpusfile,outfile):

    wordtagset = []
    with open(corpusfile,'r') as f:
        for line in f:
            wordtagset.append(json.loads(line))
    cleanedlines=[]
    for line in wordtagset:
        cleanedlines=[]
        for wordtagpair in line:
            word=wordtagpair[0]
            cleanedword=cleantheWord(word)
            cleanedlines.append(cleanedword)
        for i in range(0,len(line)):
            line[i][0] = cleanedlines[i]
    with open('unknownwordslist.txt', 'r',encoding="utf8") as f:
        unknownwordslist = f.readlines()
    
    for line in wordtagset:
        for wordtagpair in line:
            word = wordtagpair[0]
            if word in unknownwordslist:
                wordtagpair[0] = 'UNK'

    with open(outfile, 'w') as f:
        json.dump(wordtagset, f)
    return

def getcounts(unktraincorpusfile):

    with open(unktraincorpusfile,'r') as f:
        wordtagset = json.load(f)
    uniquewords = []
    for line in wordtagset:
        for wordtagpair in line:
            uniquewords.append(wordtagpair[0])
    uniquewords = list(set(uniquewords))
    with open('uniquewords.txt', 'w',encoding="utf8") as f:
        f.write(' '.join(uniquewords))

    tagset = []
    for line in wordtagset:
        for wordtagpair in line:
            tag=wordtagpair[1]
            tagset.append(tag)
            #print(tag)
    #print(tagset)
    tagset_for_counts = []
    for line in wordtagset:
        tagset_for_each_tweet=[]
        for wordtagpair in line:
            tag=wordtagpair[1]
            tagset_for_each_tweet.append(tag)
        tagset_for_counts.append(tagset_for_each_tweet)

    uni_bi_trigram_counts = countNGramFreq(tagset_for_counts, 3)

    with open('POStag_uni_bi_trigram_counts.json', 'w') as f:
        json.dump(uni_bi_trigram_counts, f)

    print('POSModel Built')
    uniquetags = list(set(tagset))

    with open('uniquetags.txt', 'w') as f:
        f.write(' '.join(uniquetags))

    wordtagcounts={}
    for line in wordtagset:
        length=len(line)
        for j in range(0,length):
            key=','.join(line[j])
            wordtagcounts.setdefault(key,0)
            wordtagcounts[key]+=1
    with open('wordtag_model.json', 'w') as f:
        json.dump(wordtagcounts, f)

    return


def Assigntags_and_check_accuracy(unkedcorpusfile, n=2):
    print("running")
    
    with open(unkedcorpusfile,'r') as f:
        wordtagset = json.load(f)

    Given_tag_sequence = []
    Predicted_tag_sequence = []
    count_for_tweets=0
    i = 0
    for line in wordtagset:
        i+= 1
        onetweet=[]
        tagsequence=[]
        for wordtagpair in line:
            onetweet.append(wordtagpair[0])
            tagsequence.append(wordtagpair[1])
        Given_tag_sequence.append(tagsequence)
        #print("giventag")
        #print(tagsequence)
        if n==2:
            if count_for_tweets==0:
                count_for_tweets+=1
                #print(onetweet)
                Predicted_tag_for_onetweet = viterbi_bigramHMM(onetweet)
                #print(Predicted_tag_for_onetweet)
                #if Predicted_tag_for_onetweet!=tagsequence:
                    #print("giventag")
                    #print(tagsequence)
                    #print(onetweet[0])
                    #print(onetweet[1])
                    #print(onetweet[2])
                    #print(Predicted_tag_for_onetweet)
            elif count_for_tweets==1:
                count_for_tweets+=1
                #print(onetweet[6])
                Predicted_tag_for_onetweet = viterbi_bigramHMM(onetweet)
                #print(Predicted_tag_for_onetweet)
                #if Predicted_tag_for_onetweet!=tagsequence:
                    #print("giventag")
                    #print(tagsequence)
                    #print(onetweet[0])
                    #print(onetweet[1])
                    #print(onetweet[2])
                    #print(Predicted_tag_for_onetweet)
            else:
                Predicted_tag_for_onetweet = viterbi_bigramHMM(onetweet)
                #print(Predicted_tag_for_onetweet)
                #if Predicted_tag_for_onetweet!=tagsequence:
                    #print("giventag")
                    #print(tagsequence)
                    #print(onetweet[0])
                    #print(onetweet[1])
                    #print(onetweet[2])
                    #print(Predicted_tag_for_onetweet)
                
                
                    

        if n==3:
            if count_for_tweets==0:
                count_for_tweets+=1
                Predicted_tag_for_onetweet = viterbi_trigramHMM(onetweet)
                #print(tagsequence)
                #print(Predicted_tag_for_onetweet)
            else:
                Predicted_tag_for_onetweet = viterbi_trigramHMM(onetweet)
                #if Predicted_tag_for_onetweet!= tagsequence:
                    #print("giventag")
                    #print(tagsequence)
                    #print(onetweet[0])
                    #print(onetweet[1])
                    #print(Predicted_tag_for_onetweet)
                    
                    

        Predicted_tag_sequence.append(Predicted_tag_for_onetweet)
    if len(Predicted_tag_sequence) != len(Given_tag_sequence):
        print("Length of predicted tag sequence does not match with the length of the provided tag sequence")
        return

    totalcount = 0
    rightcount = 0
    for each_tweet in range(0, len(Predicted_tag_sequence)):
        for tag in range(0, len(Predicted_tag_sequence[each_tweet])):
            if Predicted_tag_sequence[each_tweet][tag] == Given_tag_sequence[each_tweet][tag]:
                rightcount += 1
            totalcount += 1
    Performance_measure=(rightcount*100)/totalcount

    print("Performance Measure: ", Performance_measure)

    return

if __name__ == "__main__":
    arg = sys.argv[1]
    cleanandUNKTrainset('twt.train.json')
    cleanandUNK_dev_or_testset('twt.dev.json','dev.unk.json')
    getcounts('train.unk.json')
    
    if arg == 'test':
        print("Bigram HMM Model_test")
        Assigntags_and_check_accuracy('test.unk.json',2)
        print("Trigram HMM Model_test")
        Assigntags_and_check_accuracy('test.unk.json',3)

    elif arg == 'dev':
        print("Bigram HMM Model_dev")
        Assigntags_and_check_accuracy('dev.unk.json',2)
        print("Trigram HMM Model_dev")
        Assigntags_and_check_accuracy('dev.unk.json',3)

    elif arg == 'train':
        print("Bigram HMM Model_train")
        Assigntags_and_check_accuracy('train.unk.json',2)
        print("Trigram HMM Model_train")
        Assigntags_and_check_accuracy('train.unk.json',3)
    else:
        print('Please provide either "test" or "dev" as arguments')
    
