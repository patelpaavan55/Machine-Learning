import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    word_index={}
    gamma={}
    delta={}
    xi={}
    index=0
    
    for sentence in train_data:
        for word in sentence.words:
            if word not in word_index:
                word_index[word]=index
                index+=1

    for sentence in train_data:
        for word,tag in zip(sentence.words,sentence.tags):
            if tag not in gamma:
                gamma[tag]=1
                delta[tag]={word:1}
            else:
                gamma[tag]+=1
                if word not in delta[tag]:
                    delta[tag][word]=1
                else:
                    delta[tag][word]+=1
        for i in range(len(sentence.tags)-1):
            if sentence.tags[i] not in xi:
                xi[sentence.tags[i]]={sentence.tags[i+1]:1}
            else:
                if sentence.tags[i+1] not in xi[sentence.tags[i]]:
                    xi[sentence.tags[i]][sentence.tags[i+1]]=1
                else:
                    xi[sentence.tags[i]][sentence.tags[i+1]]+=1
    
    A=np.zeros((len(tags),len(tags)))
    B=np.zeros((len(tags),len(word_index)))
    pi=np.zeros((len(tags)))
        
    tag_index={}
    index=0
        
    for tag in tags:
        if tag not in tag_index:
            tag_index[tag]=index
        index+=1

    for tag in tags:
        if tag in gamma:
            pi[tag_index[tag]]= (gamma[tag])/(len(sentence.tags))
        
    for tag1 in tags:
        if tag1 in xi:
            for tag2 in tags:
                if tag2 in xi[tag1]:
                    A[tag_index[tag1]][tag_index[tag2]]=(xi[tag1][tag2])/sum(xi[tag1].values())
                else:
                    A[tag_index[tag1]][tag_index[tag2]]=0.0
        else:
            A[tag_index[tag1]][tag_index[tag2]]=0.0
         
    for tag1 in tags:
        if tag1 in delta:
            for tag2 in word_index:
                if tag2 in delta[tag1]:
                    B[tag_index[tag1]][word_index[tag2]]=(delta[tag1][tag2])/sum(delta[tag1].values())
                else:
                    B[tag_index[tag1]][word_index[tag2]]=0.0
        else:
            B[tag_index[tag1]][word_index[tag2]]=0.0
    
    model=HMM(pi,A,B,word_index,tag_index)
    ###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here
    
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict:
                model.obs_dict[word]=max(list(model.obs_dict.values()))+1
                model.B=np.insert(model.B, len(model.obs_dict)-1, 10**(-6), axis=1)
    
    for sentence in test_data:
        tagging.append(model.viterbi(sentence.words))  
	
    ###################################################
    return tagging
