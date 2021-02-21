import numpy as np
# ''' All data utilities are here'''


def sample_branch_data():
    branch = [[0, 4, 2], [2, 0, 4]]
    return branch


def sample_decision_tree_data():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
    labels = [0, 0, 1, 1]
    return features, labels


def sample_decision_tree_test():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c']]
    labels = [0, 0, 1]
    return features, labels

def sample_cust_data():
    features = [
    [' Yes ', ' No ', ' No ', ' Yes ', ' Some ', ' No ', ' Yes ', ' French ', '  0-10 '],
    [' Yes ', ' No ', ' No ', ' Yes ', ' Full ', ' No ', ' No ', ' Thai ', ' 30-60 '],
    [' No ', ' Yes ', ' No ', ' No ', ' Some ', ' No ', ' No ', ' Burger ', '  0-10 '],
    [' Yes ', ' No ', ' Yes ', ' Yes ', ' Full ', ' Yes ', ' No ', ' Thai ', ' 10-30 '],
    [' Yes ', ' No ', ' Yes ', ' No ', ' Full ', ' No ', ' Yes ', ' French ', ' >60 '],
    [' No ', ' Yes ', ' No ', ' Yes ', ' Some ', ' Yes ', ' Yes ', ' Italian ', '  0-10 '],
    [' No ', ' Yes ', ' No ', ' No ', ' None ', ' Yes ', ' No ', ' Burger ', '  0-10 '],
    [' No ', ' No ', ' No ', ' Yes ', ' Some ', ' Yes ', ' Yes ', ' Thai ', '  0-10 '],
    [' No ', ' Yes ', ' Yes ', ' No ', ' Full ', ' Yes ', ' No ', ' Burger ', ' >60 '],
    [' Yes ', ' Yes ', ' Yes ', ' Yes ', ' Full ', ' No ', ' Yes ', ' Italian ', ' 10-30 '],
    [' No ', ' No ', ' No ', ' No ', ' None ', ' No ', ' No ', ' Thai ', '  0-10 '],
    [' Yes ', ' Yes ', ' Yes ', ' Yes ', ' Full ', ' No ', ' No ', ' Burger ', ' 30-60 ']]
    labels=[[' Yes'], [' No'], [' Yes'], [' Yes'], [' No'], [' Yes'], 
            [' No'], [' Yes'], [' No'], [' No'], [' No'], [' Yes']]
    
    return features, labels
    


def sample_decision_tree_pruning():
    features = [[0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 2, 1, 0],
                [2, 2, 1, 1],
                [1, 2, 1, 1],
                [0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]
                ]
    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    validation = [[1, 0, 1, 1],
                  [1, 2, 0, 0],
                  [0, 1, 0, 1],
                  [0, 2, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 1],
                  [2, 0, 0, 1],
                  [2, 1, 1, 1],
                  [2, 1, 0, 1],
                  [2, 1, 0, 0]]
    v_labels = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    return features, labels, validation, v_labels


def load_decision_tree_data():
    f = open('car.data', 'r')
    white = [[int(num) for num in line.split(',')] for line in f]
    white = np.asarray(white)

    [N, d] = white.shape

    ntr = int(np.round(N * 0.66))
    ntest = N - ntr

    Xtrain = white[:ntr].T[:-1].T
    ytrain = white[:ntr].T[-1].T
    Xtest = white[-ntest:].T[:-1].T
    ytest = white[-ntest:].T[-1].T

    return Xtrain, ytrain, Xtest, ytest


def most_common(lst):
    return max(set(lst), key=lst.count)

