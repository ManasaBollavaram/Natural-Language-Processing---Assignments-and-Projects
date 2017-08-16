import re, random
import os, json
from collections import defaultdict
import math
import numpy as np

def loaddata(datafile):
    with open(datafile,'r')as f:
        words_tags_list=f.readlines()
    #print(words_tags_list[0])
    sequences_list=[]
    tag_sequences_list=[]
    sequence=[]
    word_tag_pair=[]
    tags=[]
    count=0
    for word_tag in words_tags_list:
        if word_tag.isspace():
            sequences_list.append(sequence)
            tag_sequences_list.append(tags)
            sequence=[]
            tags=[]
            count=0
        else:
            check=word_tag.split()
            #print(check)
            for word in check:
                #print(word_tag)
                count+=1
                if count<=3:
                    word_tag_pair.append(word)
                else:
                    sequence.append(word_tag_pair)
                    word_tag_pair=[]
                    tags.append(word)
                    count=0
    return sequences_list,tag_sequences_list
def global_feature_vector(sequence,tag_sequence,d):
    M=len(sequence)
    indices_global=[]
    for t in range(d):
        indices_global.append(0)
   
    indices_local_list=[]
    for k in range(0,M):
        if k==0:
            indices_local=local_feature_vector(sequence,k,tag_sequence[k],'START') #local feature vector from START to the first tag
            indices_local_list.append(indices_local)
        else:
            indices_local=local_feature_vector(sequence,k,tag_sequence[k],tag_sequence[k-1])
            indices_local_list.append(indices_local)
    for j in indices_local_list:
        for i in j:
            indices_global[i]=indices_global[i]+1
    return indices_global
def local_feature_vector(sequence,k,tag_k,tag_previousk):
    indices_local=[]
    feature_vector=[]
    feature_vector.append(sequence[k][0]+tag_k)
    feature_vector.append(sequence[k][0]+sequence[k][1]+tag_k)
    feature_vector.append(sequence[k][0]+sequence[k][2]+tag_k)
    #feature_vector.append(sequence[k][0]+sequence[k][1]+sequence[k][2]+tag_k)
    if k>0:
        if k<len(sequence)-1:
            #print(k)
            feature_vector.append(sequence[k-1][0]+sequence[k][0]+sequence[k+1][0]+tag_k)
            #feature_vector.append(sequence[k][0]+sequence[k+1][0]+tag_k)
            #feature_vector.append(sequence[k-1][0]+sequence[k][0]+tag_k)
            #feature_vector.append(sequence[k][0]+tag_previousk+tag_k)
    else:
        feature_vector.append(sequence[k][0]+tag_k+'START')
        if len(sequence)>1:
            feature_vector.append(sequence[k][0]+sequence[k+1][0]+tag_k)
    feature_vector=list(set(feature_vector))
    with open('feature_vector_traincorpus.txt','r')as f:
        compare_feature_vector=f.read().split()
    for i in feature_vector:
        for j in compare_feature_vector:
            if i==j:
                indices_local.append(compare_feature_vector.index(j))
    return indices_local
def make_feature_vector_traincorpus(trainset):
    sequences_list,tag_sequences_list=loaddata(trainset)
    M=len(sequences_list)
    feature_vector=[]
    feature=[]
    sequence=[]
    for i in range(M):
        sequence=sequences_list[i]
        tag_sequence=tag_sequences_list[i]
        for j in range(len(sequence)):
            #print(tag_sequences_list[i][j])
            feature_vector.append(sequence[j][0]+tag_sequence[j])
            feature_vector.append(sequence[j][0]+sequence[j][1]+tag_sequence[j])
            feature_vector.append(sequence[j][0]+sequence[j][2]+tag_sequence[j])
            #feature_vector.append(sequence[j][0]+sequence[j][1]+sequence[j][2]+tag_sequence[j])
            if j>0:
                if j<len(sequence)-1:
                    feature_vector.append(sequence[j-1][0]+sequence[j][0]+sequence[j+1][0]+tag_sequence[j])
                    #feature_vector.append(sequence[j][0]+sequence[j+1][0]+tag_sequence[j])
                    #feature_vector.append(sequence[j-1][0]+sequence[j][0]+tag_sequence[j])
                    #feature_vector.append(sequence[j][0]+tag_sequence[j-1]+tag_sequence[j])
            else:
                #print(j)
                feature_vector.append(sequence[j][0]+tag_sequence[j]+'START')
                if len(sequence)>1:
                    feature_vector.append(sequence[j][0]+sequence[j+1][0]+tag_sequence[j])
            #feature_vector=list(set(feature_vector))
    #feature_vector=list(set(feature_vector))
    feature_vector_dict=defaultdict(lambda: 0)
    for i in feature_vector:
        feature_vector_dict[i]+=1
    emp=[]
    for key,value in feature_vector_dict.items():
        if value>2:
            emp.append(key)
    with open('feature_vector_traincorpus.txt','w')as f:
        f.write('\n'.join(emp))
    return len(emp)
                    

def viterbi(w,sequence,unique_tags,d):
    M=len(sequence)
    N=len(unique_tags)
    back_pointers_matrix=np.ndarray(shape=(N,M), dtype=int)
    pi_previous=np.zeros(N)
    pi_current=np.zeros(N)
    pi_temp=np.zeros(N)

    #Initialization
    for i in range(0,N):
        back_pointers_matrix[i][0]=-1
        indices_local=local_feature_vector(sequence,0,unique_tags[i],'START')
        for j in indices_local:
            pi_previous[i]=pi_previous[i]+w[j]
    #print("recursion start")
    #Recursion
    for i in range(1,M):
        for j in range(N):
            for k in range(N):
                #print("local feature start")
                indices_local=local_feature_vector(sequence,i,unique_tags[j],unique_tags[k])
                #print("local feature end")
                count=0
                for p in indices_local:
                    count=count+w[p]
                pi_temp[k]=pi_previous[k]+count
            pi_previous[j]=np.amax(pi_temp)
            back_pointers_matrix[j][i]=np.argmax(pi_temp)
    back_pointer_end=back_pointers_matrix[np.argmax(pi_previous)][M-1]
    #print("recursion end")
    reverse_predicted_tag_sequence=[unique_tags[np.argmax(pi_previous)]]
    if M==1:
        return reverse_predicted_tag_sequence
    back_pointer_previous_index=np.arange(M-2,-1,-1)
    for i in back_pointer_previous_index:
        reverse_predicted_tag_sequence.append(unique_tags[back_pointer_end])
        back_pointer_end=back_pointers_matrix[back_pointer_end][i]
    predicted_tag_sequence=reverse_predicted_tag_sequence[::-1]
    return predicted_tag_sequence
  
def get_averaged_parameter_vector(datafile,d):
    sequences_list,tag_sequences_list=loaddata(datafile)
    unique_tags=['I-PER','I-ORG','I-MISC','I-LOC','O']
    T=10 #Number of iterations
    t1=np.arange(1,T,5)
    M=len(sequences_list) #M tagged sequences
    #d=100 #Number of features

    #Initialization
    w=[0]
    w_avg=[0]
    count=0
    averaged_parameter_vector=[]
    for i in range(1,d):
        w.append(0)
        w_avg.append(0)

    #Perceptron
    for t in t1:
        print(t)
        for i in range(0,M): #each_sequence
            predicted_tag_sequence=viterbi(w,sequences_list[i],unique_tags,d)
            #print('predicted')
            #print(predicted_tag_sequence)
            if predicted_tag_sequence != tag_sequences_list[i]:
                given_tag_sequence_indices=global_feature_vector(sequences_list[i],tag_sequences_list[i],d)
                predicted_tag_sequence_indices=global_feature_vector(sequences_list[i],predicted_tag_sequence,d)
                for index in range(d):
                    w[index]=w[index]+given_tag_sequence_indices[index]
                for index in range(d):
                    w[index]=w[index]-predicted_tag_sequence_indices[index]
                for j in range(len(w_avg)):
                    w_avg[j]=w_avg[j]+w[j]
                count=count+1
    for index in range(len(w_avg)):
        averaged_parameter_vector.append(w_avg[index]/count)
    with open('average_paramter_vector.txt','w') as f:
        f.write('\n'.join(averaged_parameter_vector))
    print('done')
    return averaged_parameter_vector

def Evaluate(datafile,averaged_parameter_vector,d):
    sequences_list,tag_sequences_list=loaddata(datafile)
    unique_tags=['I-PER','I-ORG','I-MISC','I-LOC','O']
    M=len(sequences_list)
    predicted_tag_sequences_list=[]
    for i in range(0,M):
        predicted_tag_sequence=viterbi(averaged_parameter_vector,sequences_list[i],unique_tags,d)
        predicted_tag_sequences_list.append(predicted_tag_sequence)
    with open('output.txt','w') as f:
        for i in range(len(sequences_list)):
            for j in range(len(sequences_list[i])):
                f.write(' '.join(sequences_list[i][j]))
                f.write(' '+tag_sequences_list[i][j]+' '+predicted_tag_sequences_list[i][j]+'\n')
            f.write('\n')
        
        
    
if __name__ == "__main__":
    arg = sys.argv[1]
    
    #d=Number of features
    d=make_feature_vector_traincorpus('eng.train.small')
    with open('feature_vector_traincorpus.txt','r') as f:
        p=f.read().split()
    d=len(p)
    print(d)
    averaged_parameter_vector=get_averaged_parameter_vector('eng.train.small',d)
    
    
    if arg == 'test':
        print("Perceptron for test set")
        Evaluate('eng.test.small',averaged_parameter_vector,d)

    elif arg == 'dev':
        print("Perceptron for dev set")
        Evaluate('eng.dev.small',averaged_parameter_vector,d)
        

    elif arg == 'train':
        print("Perceptron for train corpus")
        Evaluate('eng.train.small',averaged_parameter_vector,d)
        
    else:
        print('Please provide either "test" or "dev" or "train" as arguments')
    

    
    
