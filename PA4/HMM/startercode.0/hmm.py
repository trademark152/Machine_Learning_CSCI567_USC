from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        # loop through each state
        # for state in range(S):
        #     # calculate initial transitional probability: initial state emits initial observation delta= Pi*B
        #     # use obs_dict to get the index of observation in Osequence[0]
        #     alpha[state, 0] = self.pi[state] * self.B[state, self.obs_dict[Osequence[0]]]
        # print("Osequence: ", Osequence)
        alpha[:, 0] = np.multiply(self.pi, self.B[:, self.obs_dict[Osequence[0]]])

        # loop through each time step starting at 1 (0 has been calculated)
        for time in range(1, L):
            # loop through each starting state
            # for endState in range(S):
            #     # calculate transitional probability from start state to end state
            #     alpha[endState, time] = self.B[endState, self.obs_dict[Osequence[time]]] * sum(
            #         [self.A[startState, endState] * alpha[startState, time - 1] for startState in range(S)])
            
            # more efficient way of computing:
            alpha[:, time] = np.multiply(self.B[:, self.obs_dict[Osequence[time]]], sum(
                [np.multiply(self.A[startState, :], alpha[startState, time - 1]) for startState in range(S)]))
        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        # loop through each state
        # for state in range(S):
        #     beta[state, L - 1] = 1
        beta[:, L - 1] = 1

        for time in reversed(range(L - 1)):
            # for startState in range(S):
            #     beta[startState, time] = sum([beta[endState, time + 1] * self.A[startState, endState] * self.B[
            #         endState, self.obs_dict[Osequence[time + 1]]] for endState in range(S)])

            # more efficient way of computing
            beta[:, time] = sum([beta[endState, time + 1] * self.A[:, endState] * self.B[
                endState, self.obs_dict[Osequence[time + 1]]] for endState in range(S)])
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)

        beta = self.backward(Osequence)

        # probability of observing the sequence (slide 33 lecture 11)
        # take sum of all columns of the resultant numState*L matrices
        # pick any time marker
        prob = np.multiply(alpha, beta).sum(axis=0)[0]
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)

        beta = self.backward(Osequence)

        constant = self.sequence_prob(Osequence)

        prob = np.divide(np.multiply(alpha, beta), constant)
        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        S = len(self.pi)
        L = len(Osequence)

        """ pointer : numpy.ndarray SAME AS BIG DELTA in slides
            Contains a pointer to the previous state at t-1 that maximizes
            delta[state][t]
        """
        pointer = np.zeros((S, L-1), dtype=int)

        # matrix containing max likelihood of state at a given time
        """         delta : numpy.ndarray
            delta [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        """
        delta = np.zeros((S, L))
        # in case unseen words out of test_data is encountered
        if Osequence[0] not in self.obs_dict:
            delta[:, 0] = self.pi
        else:
            delta[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        for time in range(1, L):
            for state in range(S):
                if Osequence[time] not in self.obs_dict:
                    seq_probs = delta[:, time - 1] * self.A[:, state]
                else:
                    seq_probs = delta[:, time-1] * self.A[:, state] * self.B[state, self.obs_dict[Osequence[time]]]
                pointer[state, time-1] = np.argmax(seq_probs)
                delta[state, time] = np.max(seq_probs)

        # print("delta ", delta)
        # print("pointer ", pointer)

        # obtain starting point of inverted final path from delta matrix: starting backward from final time
        lastState = np.argmax(delta[:, L-1])
        path.insert(0, lastState)
        # print("last state: ", path)

        # invert final path
        for time in range(L - 1):
            # invert the time to go backwards
            time = L - 1 - time

            # get the previous best state from pointer matrix
            previousState = pointer[lastState, time-1]

            # add to path
            path.insert(0, previousState)

            # update last state
            lastState = previousState

        # print("inverted path: ", path)

        # return state not indexes
        # print(self.state_dict)
        finalPath = []
        for point in path:
            for state, idx in self.state_dict.items():  # for name, age in dictionary.iteritems():
                if idx == point:
                    finalPath.append(state)
        path = finalPath

        ###################################################
        return path
