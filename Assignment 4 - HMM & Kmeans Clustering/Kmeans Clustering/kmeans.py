import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
#    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
#    first=generator.choice(n,size=1)
#    C=[first[0]]
#    for k in range(1,n_cluster):
#        D2 = np.array([np.amin([np.linalg.norm(np.array(c)-np.array(xi),np.array(c)-np.array(xi)) for c in C]) for xi in range(len(x))])
#        probs= D2/D2.sum()
#        i=np.argmax(probs)
#        cumprobs = probs.cumsum()
#        i=-1
#        r=np.random.rand()
#        for j,p in enumerate(cumprobs):
#            if r<p:
#                i = j
#                break
#        C.append(i)
#    centers=C
#    
    
    centers=[]
    first_center=generator.choice(n, size=1)
    centers.append(first_center[0])
    for i in range(1, n_cluster):
        minimum_distance=dict()
        for dp in range(len(x)):
            if dp not in centers:
                min_dist=[]
                for c in centers:
                    min_dist.append(np.linalg.norm(c-dp))
                minimum_distance[dp]=np.amin(min_dist)
        index=max(minimum_distance,key=minimum_distance.get)
        centers.append(index)
            
            
#    raise Exception(
#             'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeans class')
#        centroids=x[self.centers]
#        centroids_old=np.zeros(centroids.shape)
#        membership=np.zeros((N,))
#        dist=np.linalg.norm(centroids-centroids_old)
#        iteration=0
#        while dist>self.e :#and iteration<self.max_iter
#            iteration +=1
#            dist=np.linalg.norm(centroids-centroids_old)
#            for idx_inst, inst in enumerate(x):
#                dist_vec = np.zeros((self.n_cluster,1))
#                for idx_centroid, centroid in enumerate(centroids):
#                    dist_vec[idx_centroid] = np.linalg.norm(centroid-inst)
#                membership[idx_inst,]=np.argmin(dist_vec)
#        
#            tmp_centroids=np.zeros((self.n_cluster,D))
#            
#            for centroid_idx in range(len(centroids)):
#                cluster_insts=[i for i in range(len(membership)) if membership[i]==centroid_idx]
#                centroid = np.mean(x[cluster_insts],axis=0)
#                tmp_centroids[centroid_idx,] = centroid
#            
#            centroids= tmp_centroids
#        y=membership
#        return centroids, y, iteration
        centroids=x[self.centers]
#        J=10**10
        # mu - centroids
        # c - centroids assigned to each in vector vec
        # vec - vector data points
        #def distortion(mu, c, vec):
        #return ((mu[c] - vec) ** 2).sum() / vec.shape[0]
        for i in range(self.max_iter):        
            
#            distances = np.sqrt(((x-centroids[:,np.newaxis])**2).sum(axis=2))
#            closest=np.argmin(distances,axis=0)
            closest=np.argmin(((x[:,:,None]-centroids.T[None,:,:])**2).sum(axis=1),axis=1)
            new_centroids=np.array([x[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
            if abs(centroids.sum()-new_centroids.sum())<=self.e:
                break
            else:
                centroids = new_centroids
#                J=J_new
#        print(abs(new_centroids-centroids))
        return centroids,closest,i+1
        
#        print("iterations:",iteration)
        
        # DO NOT CHANGE CODE BELOW THIS LINE
#        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeansClassifier class')
        centroids=x[self.centers]
        for i in range(self.max_iter):        
            
#            distances = np.sqrt(((x-centroids[:,np.newaxis])**2).sum(axis=2))
#            closest=np.argmin(distances,axis=0)
            closest=np.argmin(((x[:,:,None]-centroids.T[None,:,:])**2).sum(axis=1),axis=1)
            new_centroids=np.array([x[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
            
            if (new_centroids==centroids).all():
                break
            else:
                centroids = new_centroids
            
        centroid_labels=np.array([np.argmax(np.bincount(y[closest==k])) for k in range(centroids.shape[0])])
        #Not incorporated condition 2 

        

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement predict function in KMeansClassifier class')
        closest=np.argmin(((x[:,:,None]-self.centroids.T[None,:,:])**2).sum(axis=1),axis=1)
        labels=self.centroid_labels[closest]
        
        return labels
        # DO NOT CHANGE CODE BELOW THIS LINE
#        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
#    raise Exception(
#             'Implement transform_image function')
    N,M = image.shape[:2]
    flat_image=image.reshape(-1,3)
    closest_vector=np.argmin(((flat_image[:,:,None]-code_vectors.T[None,:,:])**2).sum(axis=1),axis=1)#closest_vector->(N*M,1) numpy array
    new_im=code_vectors[closest_vector]#new_im->(N*M,3) numpy array
    new_im=new_im.reshape(N,M,3)
#    closest=np.argmin(((x[:,:,None]-self.centroids.T[None,:,:])**2).sum(axis=1),axis=1)
#    centroid_labels=np.array([np.argmax(np.bincount(y[closest==k])) for k in range(centroids.shape[0])])
#    labels=self.centroid_labels[closest]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

