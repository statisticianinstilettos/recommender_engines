import numpy as np

class ContextualMultiArmedBandit():
    def __init__(self, m):
        self.agent
        self.observation
        self.epsilon #the constantly adapting explore/exploit ratio

    def action(self, user_abc):
        #recommend a book 
        #The rec should be based on predicted rating probability given book and user content
        return book_abc
    
    def reward(book_abc, userabc):
        #get the reward for the action taken

    def update_observation(self, x):
        #increment N to take a step forward in time
        self.N += 1
        #update the observation of the environment with the new information.
        self.observation = #some stuff
        
data is like a tuple {user:{'abc'}, item:{'123'}, rating={'xyz'}}
or generalize it out to
data is like a tuple {agent:{'abc'}, action:{'123'}, reward={'xyz'}, observation:{'state at time N'}, time:{N}, regreg:{123}}

class UCBContextualBandit(ContextualMultiArmedBandit):
    def __init__(self, m):
        self.mean = 0 #initialized the mean of the bandits
        super().__init__(m)
    #psudocode. waht this going to do?
    
    #select the best actin for a given agent usign the UCB method.
    #observe and record the outcome
    #update the state
    #retrain the ML algo with new after every x actions. (500, 1000?) Make this configurage. 
    #check in with regret along the way to see if the model is learning. 
    #watch how explore/exploit ration changes and compare to uncertaintly and probabaility of highpositive reward to make sure shit is working
    
    #big someday idea
    #introduce bias and see if you can simulate a biased and unfair system that self corrects itsself in this framework. 
    
        
        
         
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
    
