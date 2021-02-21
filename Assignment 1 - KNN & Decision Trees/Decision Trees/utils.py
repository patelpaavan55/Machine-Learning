import numpy as np
import copy


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    total_data_points=0.0
    attr_tot_arr=[]
    for at_i in range(len(branches)):
        att_tot=0
        for point in branches[at_i]:
            att_tot+=point
            total_data_points+=point
        attr_tot_arr.append(att_tot)

    weighted_entropy=0.0
    for at_i in range(len(branches)):
        entropy=0
        for point in branches[at_i]:
            if point!=0:
                x=point/attr_tot_arr[at_i]
                entropy=entropy-x*np.log2(x)
        weighted_entropy=weighted_entropy+(attr_tot_arr[at_i]/total_data_points)*entropy
    ig=S-weighted_entropy
    return ig
#    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    prune_node(decisionTree, decisionTree.root_node, X_test, y_test)
#    raise NotImplementedError

def prune_node(decisionTree, node, x_test, y_test):
    if(node.splittable==False) or len(node.children)==0:
        return
    no_of_children=len(node.children)
    
    for i in range(no_of_children):
        prune_node(decisionTree, node.children[i], x_test, y_test)
    
#    tempX=np.array(x_test,copy=True).tolist()
    tempX=copy.deepcopy(x_test)
    accuracy=accuracy_score(decisionTree.predict(tempX),y_test)
    
    children=node.children
    node.children=[]
    node.splittable=False
    
    new_tempX=copy.deepcopy(x_test)
    new_accuracy=accuracy_score(decisionTree.predict(new_tempX),y_test)
    
    if new_accuracy < accuracy:
        node.children=children
        node.splittable=True
    return
    
def accuracy_score(arr_pred, arr_true):
    no_of_matches=0
    for i in range(len(arr_pred)):
        if arr_pred[i]==arr_true[i]:
            no_of_matches+=1
    accuracy=no_of_matches/len(arr_pred)
    return accuracy

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
