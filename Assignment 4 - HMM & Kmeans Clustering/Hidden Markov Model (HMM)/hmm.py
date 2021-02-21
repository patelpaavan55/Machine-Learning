from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        for i in range(S):
            alpha[i][0]=self.pi[i]*self.B[i][self.obs_dict[Osequence[0]]]
        for i in range(1,L):
            for j in range(S):
                for k in range(S):
                    alpha[j][i]+=self.B[j][self.obs_dict[Osequence[i]]]*self.A[k][j]*alpha[k][i-1]
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        for i in range(S):
            beta[i][L-1]=1
        for i in range(L-2,-1,-1):
            for j in range(S):
                for k in range(S):
                    beta[j][i]+=self.A[j][k]*self.B[k][self.obs_dict[Osequence[i+1]]]*beta[k][i+1]
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        arr=self.forward(Osequence)
        for i in range(len(self.pi)):
            prob+=arr[i][len(Osequence)-1]
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        forward_array=self.forward(Osequence)
        backward_array=self.backward(Osequence)
        seq_prob=self.sequence_prob(Osequence)
        for i in range(len(self.pi)):
            for j in range(len(Osequence)):
                prob[i][j]=(forward_array[i][j]*backward_array[i][j])/seq_prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        forward_array=self.forward(Osequence)
        backward_array=self.backward(Osequence)
        seq_prob=self.sequence_prob(Osequence)
        for i in range(L-1):
            for j in range(S):
                for k in range(S):
                    prob[j][k][i]=(self.A[j][k]*self.B[k][self.obs_dict[Osequence[i+1]]]*backward_array[k][i+1]*forward_array[j][i])/seq_prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        state_dict_inverse = {v: k for k, v in self.state_dict.items()}
        S = len(self.pi)
        L = len(Osequence)
        temp_path = [0]*L
        delta = np.zeros([S,L])
        backtrack_matrix = np.zeros([S,L])
        
        for i in range(S):
            delta[i][0]=self.pi[i]*self.B[i][self.obs_dict[Osequence[0]]]
        for i in range(1,L):
            for j in range(S):
                max_val=-1000000
                max_index=-1
                for k in range(S):
                    if max_val < self.B[j][self.obs_dict[Osequence[i]]]*self.A[k][j]*delta[k][i-1]:
                        max_val = self.B[j][self.obs_dict[Osequence[i]]]*self.A[k][j]*delta[k][i-1]
                        max_index = k
                delta[j][i]=max_val
                backtrack_matrix[j][i]=max_index
                
        max_val=-1000000
        for i in range(S):
            if max_val < delta[i][L-1]:
                max_val = delta[i][L-1]
                max_index = i
        temp_path[L-1]=max_index
        
        for i in range(L-2,-1,-1):
            temp_path[i]=backtrack_matrix[int(temp_path[i+1])][i+1]
        
        for i in range(len(temp_path)):
            path.append(state_dict_inverse[temp_path[i]])
            
        ###################################################
        return path
