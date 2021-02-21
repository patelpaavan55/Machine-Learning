import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    #y->real labels  y^-> predicted label
    assert len(real_labels) == len(predicted_labels)
    for i in range(len(real_labels)):
        real_labels[i]=float(real_labels[i])
        predicted_labels[i]=float(predicted_labels[i])
    numr=0
    den=0
    for i in range(len(real_labels)):
        numr=numr+(real_labels[i]*predicted_labels[i])
        den=den+real_labels[i]+predicted_labels[i]
    f1=float((2*numr)/den)
    return f1
    
    
    #raise NotImplementedError
#y_true = [0, 1, 1, 0, 1, 0]
#y_pred = [0, 0, 1, 0, 0, 1]
#y_true = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]
#y_pred = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
#y_true = [0, 1, 2, 0, 1, 2]
#y_pred = [0, 2, 1, 0, 0, 1]
#print(f1_score(y_true, y_pred))

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dist=0
        for i in range(len(point1)):
            dist=dist+(abs(point1[i]-point2[i]))** (3)
        mink_dist=dist ** (1./3)
        return mink_dist
#        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dist=0
        for i in range(len(point1)):
            dist=dist+(point1[i]-point2[i])** (2)
        euc_dist=dist ** (1./2)
        return euc_dist
#        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        dist=0
        for i in range(len(point1)):
            dist=dist+(point1[i]*point2[i])
        return dist
#        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dist=0.0
        mod1=0.0
        mod2=0.0
        for i in range(len(point1)):
            dist=dist+(point1[i]*point2[i])
            mod1=mod1+(point1[i]**2)
            mod2=mod2+(point2[i]**2)
        mod1=np.sqrt(mod1)
        mod2=np.sqrt(mod2)
        m1m2=mod1*mod2
        if m1m2==0:
            m1m2=0.0000000001
        cos_sim=(dist/m1m2)
        cos_dist=float(1-cos_sim)
        return cos_dist
#        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dist=0
        for i in range(len(point1)):
            dist=dist+((point1[i]-point2[i])**2)
        gaus_dist=-np.exp((-1./2)*dist)
        return gaus_dist
#        raise NotImplementedError


#vector1 = [0, 2, 3, 4] 
#vector2 = [2, 4, 3, 7]
#print(Distances().minkowski_distance(vector1,vector2))
#v1=[1,2,3]
#v2=[0,1,0]
#print(Distances().inner_product_distance(v1,v2))

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def priority(self, arr):
        if 'euclidean' in arr:
            return 'euclidean'
        elif 'minkowski' in arr:
            return 'minkowski'
        elif 'gaussian' in arr:
            return 'gaussian'
        elif 'inner_prod' in arr:
            return 'inner_prod'
        elif 'cosine_dist' in arr:
            return 'cosine_dist'

    def encode(self, arr):
        if 'euclidean' in arr:
            return 1
        elif 'minkowski' in arr:
            return 2
        elif 'gaussian' in arr:
            return 3
        elif 'inner_prod' in arr:
            return 4
        elif 'cosine_dist':
            return 5
    
    def decode(self, arr):
        if 1==arr:
            return 'euclidean'
        elif 2==arr:
            return 'minkowski'
        elif 3==arr:
            return 'gaussian'
        elif 4==arr:
            return 'inner_prod'
        elif 5==arr:
            return 'cosine_dist'
        
    def sc_decode(self, val):
        if 0==val:
            return 'min_max_scale'
        elif 1==val:
            return 'normalize'
    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
#        distance_funcs = {
#                'euclidean': Distances.euclidean_distance,
#                'minkowski': Distances.minkowski_distance,
#                'gaussian': Distances.gaussian_kernel_distance,
#                'inner_prod': Distances.inner_product_distance,
#                'cosine_dist': Distances.cosine_similarity_distance,
#                }
        
        best=[]
        for k in range(1,30,2):
            f1_arr=[]
            for dd in distance_funcs:
                temp=KNN(k,distance_funcs[dd])
                temp.train(x_train, y_train)
                x1=temp.predict(x_val)
                f1_arr.append([dd,f1_score(y_val,x1),temp])
            new_f1=sorted(f1_arr, key= lambda x: x[1], reverse=True)
            mx_f=new_f1[0][1]
            mx_model=new_f1[0][2]
#            for el in new_f1:
#                print(el,"---", el[0],"---", el[1])
            mx_dd=[]
            for q in new_f1:
                if q[1]==mx_f:
                    mx_dd.append(q[0])
#            new_f1=sorted(sorted(f1_arr, key = lambda x : x[0]), key = lambda x : x[1], reverse = True)
#            print("T",new_f1[0]," key:",dd," s",new_f1[1])
            xx=[k,self.priority(mx_dd),mx_f,mx_model]
            best.append(xx)
        for i in best:
            i[1]=self.encode(i[1])
#            print(i)
#        best_sorted=sorted(best,key= lambda x: x[2], reverse=True)
#        bst_f1=best_sorted[0][2]
        bst_of_bst=sorted(sorted(sorted(best, key = lambda x : x[0]), key = lambda x : x[1]), key = lambda x: x[2], reverse=True)
#        print("Bst of Best")
#        for i in bst_of_bst:
#            print(i)
        fnl=bst_of_bst[0]
#        print("Best K, Best Dist_function, Best_F: {0}, {1}, {2}".format(fnl[0], self.decode(fnl[1]), fnl[2]))
        
        
        
        
#        for el in f1_arr:
#            print el
            
#        for e1,e2 in zip(f1_arr,new_f1):
#            print("old: ",e1," new: ",e2)
            
        # You need to assign the final values to these variables
        self.best_k = fnl[0]
        self.best_distance_function = self.decode(fnl[1])
        self.best_model = fnl[3]
#        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and distance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
#        distance_funcs = {
#                'euclidean': Distances.euclidean_distance,
#                'minkowski': Distances.minkowski_distance,
#                'gaussian': Distances.gaussian_kernel_distance,
#                'inner_prod': Distances.inner_product_distance,
#                'cosine_dist': Distances.cosine_similarity_distance,
#                }
#        scaling_classes = {
#            'min_max_scale': MinMaxScaler,
#            'normalize': NormalizationScaler,
#            }
        best=[]
        for k in range(1,30,2):
            f1_arr=[]
            for dd in distance_funcs:
                
                sc_f1=[]
                sc_mod=[]
                for sc in scaling_classes:
                    temp=KNN(k,distance_funcs[dd])
                    norm_xtrain_obj=scaling_classes[sc]()
                    norm_xtrain=norm_xtrain_obj(x_train)
                    temp.train(norm_xtrain, y_train)
#                    norm_xval_obj=scaling_classes[sc]()
                    norm_xval=norm_xtrain_obj(x_val)
                    y_pred=temp.predict(norm_xval)
                    sc_f1.append(f1_score(y_val,y_pred))
                    sc_mod.append(temp)
                sc_mx=max(sc_f1)
                sc_indx=[]
                for i in range(len(sc_f1)):
                    if sc_f1[i]==sc_mx:
                        sc_indx.append(i)
                if len(sc_indx)==1:
                    f1_arr.append([dd,sc_mx,sc_indx[0],sc_mod[sc_indx[0]]])
                else:
                    f1_arr.append([dd,sc_mx,0,sc_mod[0]])   
#                f1_arr.append([dd,f1_score(y_val,x1)])
            new_f1=sorted(f1_arr, key= lambda x: x[1], reverse=True)
#            print("New_f1",new_f1)
            mx_f=new_f1[0][1]
            mm_sc=new_f1[0][2]
            mx_mod=new_f1[0][3]
            mx_dd=[]
            for q in new_f1:
                if q[1]==mx_f:
                    mx_dd.append(q[0])
            xx=[k,self.priority(mx_dd),mx_f,mm_sc,mx_mod]
            best.append(xx)
        for i in best:
            i[1]=self.encode(i[1])
        bst_of_bst=sorted(sorted(sorted(best, key = lambda x : x[0]), key = lambda x : x[1]), key = lambda x: x[2], reverse=True)
        fnl=bst_of_bst[0]        
        # You need to assign the final values to these variables
        self.best_k = fnl[0]
        self.best_distance_function = self.decode(fnl[1])
        self.best_model = fnl[4]
#        print(fnl[3],self.sc_decode(fnl[3]))
        self.best_scaler =self.sc_decode(fnl[3])
        print("F1:",fnl[2])

#        raise NotImplementedError
#        self.best_k = fnl[0]
#        self.best_distance_function = self.decode(fnl[1])
#        self.best_model = fnl[4]
##        print(fnl[3],self.sc_decode(fnl[3]))
#        self.best_scaler =self.sc_decode(fnl[3])


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm_features=[]
        for f in features:
            modp=0.0
            for p in f:
                modp=modp+(p*p)
            modp=np.sqrt(modp)
            if modp!=0:
                norm_f=[x/modp for x in f]
            else:
                norm_f=f
            norm_features.append(norm_f)
        return norm_features
#        raise NotImplementedError

#features = [[3, 4], [1, -1], [0, 0]]
#asd=NormalizationScaler()
#print(asd(features))

class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first=True
        self.max_arr=0.0
        self.min_arr=0.0
#        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        numpy_feat=np.array([np.array(xi) for xi in features])
        numpy_feat=numpy_feat.astype(float)
        temp_num=np.zeros_like(numpy_feat)
        if(self.first):
            self.max_arr=numpy_feat.max(axis=0)
            self.min_arr=numpy_feat.min(axis=0)
            self.first=False

        rows=numpy_feat.shape[0]
        cols=numpy_feat.shape[1]
#        print(features)
#        print(self.max_arr)
#        print(self.min_arr)
        for i in range(0, rows):
            for j in range(0, cols):
#                print(numpy_feat[i,j])
                deno=(self.max_arr[j]-self.min_arr[j])
                if deno!=0:
                    temp_num[i,j]=(numpy_feat[i,j]-self.min_arr[j])/deno
                else:
                    temp_num[i,j]=1
        return temp_num.tolist()
            
#        raise NotImplementedError

#qw=MinMaxScaler()
#
#qw([[2, -1], [-1, 5], [0, 0]])
#qw([[1, 2], [3, 4], [5, 6]])
