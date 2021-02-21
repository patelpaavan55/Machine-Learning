import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split
        
#    def transform(self):
#        self.numpy_features=np.array([np.array(xi) for xi in self.features])
#        np_feat=self.numpy_features
#        
##        dict_features={}
#        rows = self.numpy_features.shape[0]
#        cols = self.numpy_features.shape[1]
#        unique_attr_vals=np.array([np.unique(np_feat[:,j]) for j in range(0,cols)])
#        print(unique_attr_vals)
#        for x in range(0, rows):
#            for y in range(0, cols):
#                if dict_features.get(np_feat[x]) ==None:
#                    dict_features[np_feat[x]]

    #TODO: try to split current node
    def split(self):
        if self.splittable == False:
            return

        if len(self.features[0]) == 0:
            self.splittable=False
            return

        if len(self.features) == 0:
            return
        
        igmax=-1
        more_attributes=-1
        S=0.0
        ulabels=np.unique(self.labels)
        for l in range(len(ulabels)):
                if self.labels.count(ulabels[l])==0:
                    S+=0
                    continue
                frac = self.labels.count(ulabels[l]) / (len(self.labels))
                S -= frac * np.log2(frac)
        igSum=0.0
        for i in range(len(self.features[0])):
            ig=0
            feature_column=[row[i] for row in self.features]
            
            labels=self.labels     
            unique_features=np.unique(feature_column)
            unique_labels=np.unique(labels)
            
            dict_features={}
            dict_labels={}

            for k in range(len(unique_features)):
                dict_features[unique_features[k]]=k

            for p in range(len(unique_labels)):
                dict_labels[unique_labels[p]]=p


            branches=[[0 for x in range(len(unique_labels.tolist()))] for y in range(len(unique_features.tolist()))]

            for l in range(len(feature_column)):
                if feature_column[l] in dict_features:
                    branches[dict_features[feature_column[l]]][dict_labels[labels[l]]]+=1

            ig=Util.Information_Gain(S,branches)
            igSum+=ig

            if ig>igmax:
                igmax=ig
                self.dim_split=i
                self.feature_uniq_split=unique_features
                more_attributes=len(np.unique(unique_features))
                
            elif ig==igmax:
                if len(np.unique(unique_features))>more_attributes:
                    igmax=ig
                    self.dim_split=i
                    self.feature_uniq_split=unique_features
                    more_attributes=len(np.unique(unique_features))

        if igSum==0:
            self.splittable=False
            return
        else:
            feature_column=[row[self.dim_split] for row in self.features]
            unique_features = self.feature_uniq_split
            
            for i in range(len(unique_features)):
    
                new_features_list=[]
                new_labels_list=[]
    
                for j in range(len(self.features)):
                    if unique_features[i]==feature_column[j]:
                        new_features_list.append(self.features[j])
                        new_labels_list.append(self.labels[j])
                
                new_features_list=np.asarray(new_features_list)
                x1=new_features_list.transpose()
                x2=np.delete(x1,self.dim_split,0)
                x3=x2.transpose().tolist()
                new_features_list=x3
    #            new_features_list=(np.delete(new_features_list.transpose(),self.dim_split,0)).transpose().tolist()
                
                
                child=TreeNode(new_features_list,new_labels_list,len(np.unique(new_labels_list)))
                if len(new_features_list)==0:
                    child.cls_max=self.cls_max
                    child.splittable=False       
                if len(new_features_list[0])==0:
                    count_max=0
                    for label in np.unique(new_labels_list):
                        if new_labels_list.count(label) > count_max:
                            count_max = new_labels_list.count(label)
                            child.cls_max = label
                    child.splittable=False
                self.children.append(child)
                
            for i in range(len(unique_features)):
                self.children[i].split() 
                
#        raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if len(feature)==0:
            return self.cls_max
        
        if self.splittable==False:
            return self.cls_max
        
        value=feature[self.dim_split]
        child_index=-1
        
        for i in range(len(self.children)):
            if self.feature_uniq_split[i]==value:
                child_index=i
                break
            
        feature.pop(self.dim_split)
        if child_index==-1:
            return self.cls_max
        return self.children[child_index].predict(feature)
#        raise NotImplementedError

#features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
#labels = [0, 0, 1, 1]
#x=TreeNode(features,labels,2)
#x.transform()