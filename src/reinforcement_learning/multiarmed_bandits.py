import numpy as np

class MultiArmedBandit():
    def __init__(self, m):
        self.m = m #true mean of the bandit
        self.N = 0 #number of trials

    def pull(self):
        #return reward of pulling the bandit. 
        #Calculated as a sample from the standard normal, plus the mean of the bandit
        return np.random.randn() + self.m 

    def update(self, x):
        #increment N
        self.N += 1
        #update mean of the bandit
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x
        
class GreedyBandit(MultiArmedBandit):
    def __init__(self, m):
        self.mean = 0 #initialized the mean of the bandits
        super().__init__(m)
    
    def perform_epsilon_greedy(bandit_true_means, N, epsilon):

        bandits = [GreedyBandit(x) for x in bandit_true_means]

        for i in range(N):

            #use epsilon greedy to seslect action
            p = np.random.random()
            if p < epsilon:
                #randomly select an action
                j = np.random.choice(3)
            else:
                #select action with current best mean
                j = np.argmax([b.mean for b in bandits])

            #update mean of bandit pulled
            x = bandits[j].pull()
            bandits[j].update(x)

        return bandits

class OptimisticBandit(MultiArmedBandit):
    def __init__(self, upper_limit, m):
        self.mean = upper_limit #initialized the mean of the bandits
        super().__init__(m)
        
    def perform_optimistic_initialization(bandit_true_means, N, upper_limit):
        
        bandits = [OptimisticBandit(x, y) for x,y in zip(bandit_true_means, [upper_limit]*len(bandit_true_means))]

        for i in range(N):

            #pick bandit with highest mean
            j = np.argmax([b.mean for b in bandits])

            #update mean of bandit pulled
            x = bandits[j].pull()
            bandits[j].update(x)
            
        return bandits

class UCB1Bandit(MultiArmedBandit):
    def __init__(self, m):
        self.mean = 0 #initialized the mean of the bandits
        super().__init__(m)
         
    def perform_ucb1(bandit_true_means, N, upper_limit):
        
        def ucb(mean, n, nj):
            if nj == 0:
                return float('inf')
            return mean + np.sqrt(2*np.log(n) / nj)
        
        bandits = [UCB1Bandit(x) for x in bandit_true_means]

        for i in range(N):

            #pick bandit with highest mean
            j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])

            #update mean of bandit pulled
            x = bandits[j].pull()
            bandits[j].update(x)
            
        return bandits
    
class BayesianBandit(MultiArmedBandit):
    def __init__(self, m):
        #parameters for mu - prior is N(0,1)
        self.mean = 0
        self.lambda_ = 1
        self.sum_x = 0 #for convenience
        self.tau = 1
        super().__init__(m)

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.mean = self.tau*self.sum_x / self.lambda_
        
    def perform_bayesian_bandits(bandit_true_means, N):
        
        bandits = [BayesianBandit(x) for x in bandit_true_means]

        for i in range(N):

            #pick bandit with highest mean
            j = np.argmax([b.mean for b in bandits])

            #update mean of bandit pulled
            x = bandits[j].pull()
            bandits[j].update(x)
            
        return bandits