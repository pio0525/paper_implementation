import math
import random
import pandas as pd
import numpy as np

def ind_max(x):
  m = max(x)
  return x.index(m)

def categorical_draw(probs):
  z = random.random()
  cum_prob = 0.0
  for i in range(len(probs)):
    prob = probs[i]
    cum_prob += prob
    if cum_prob > z:
      return i
  
  return len(probs) - 1


# EpsilonGreedy algorithm
class EpsilonGreedy():
  def __init__(self, epsilon, counts, values):
    self.epsilon = epsilon
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    if random.random() > self.epsilon:
      return ind_max(self.values)
    else:
      return random.randrange(len(self.values))
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

  def run_simul(self, reward) : 
    n_observation = reward.shape[0]; n_arms = reward.shape[1]
    
    rec_reward_get = []; rec_reward_max = []; rec_correct =[]
    for i in range(n_observation) : 
        # choose action
        chosen_arm = self.select_arm()
        best_arm = reward[i].argmax()
        
        # observe and recored outcomes
        reward_get = reward[i,chosen_arm]
        reward_max = max(reward[i])
        

        if chosen_arm == best_arm :
            rec_correct.append(1)
        else :
            rec_correct.append(0)
        
        rec_reward_get.append(reward_get)
        rec_reward_max.append(reward_max)
        
        # update algorithm
        self.update(chosen_arm = chosen_arm, reward = reward_get)
    
    simul_result = pd.DataFrame({'reward':rec_reward_get, 'max':rec_reward_max, 'correct':rec_correct})
    simul_result['regret'] = simul_result['max'] - simul_result['reward']

    return(simul_result)
        
        
        

# Annealing EpsilonGreedy algorithm
class AnnealingEpsilonGreedy():
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def select_arm(self):
    t = sum(self.counts) + 1
    epsilon = 1 / math.log(t + 0.0000001)
    
    if random.random() > epsilon:
      return ind_max(self.values)
    else:
      return random.randrange(len(self.values))
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

  def run_simul(self, reward) : 
    n_observation = reward.shape[0]; n_arms = reward.shape[1]
    
    rec_reward_get = []; rec_reward_max = []; rec_correct =[]
    for i in range(n_observation) : 
        # choose action
        chosen_arm = self.select_arm()
        best_arm = reward[i].argmax()
        
        # observe and recored outcomes
        reward_get = reward[i,chosen_arm]
        reward_max = max(reward[i])
        

        if chosen_arm == best_arm :
            rec_correct.append(1)
        else :
            rec_correct.append(0)
        
        rec_reward_get.append(reward_get)
        rec_reward_max.append(reward_max)
        
        # update algorithm
        self.update(chosen_arm = chosen_arm, reward = reward_get)
    
    simul_result = pd.DataFrame({'reward':rec_reward_get, 'max':rec_reward_max, 'correct':rec_correct})
    simul_result['regret'] = simul_result['max'] - simul_result['reward']

    return(simul_result)


# Softmax algorithm
class Softmax:
  def __init__(self, temperature, counts, values):
    self.temperature = temperature
    self.counts = counts
    self.values = values
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return
  
  def select_arm(self):
    z = sum([math.exp(v / self.temperature) for v in self.values])
    probs = [math.exp(v / self.temperature) / z for v in self.values]
    return categorical_draw(probs)

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

  def run_simul(self, reward) : 
    n_observation = reward.shape[0]; n_arms = reward.shape[1]
    
    rec_reward_get = []; rec_reward_max = []; rec_correct =[]
    for i in range(n_observation) : 
        # choose action
        chosen_arm = self.select_arm()
        best_arm = reward[i].argmax()
        
        # observe and recored outcomes
        reward_get = reward[i,chosen_arm]
        reward_max = max(reward[i])
        

        if chosen_arm == best_arm :
            rec_correct.append(1)
        else :
            rec_correct.append(0)
        
        rec_reward_get.append(reward_get)
        rec_reward_max.append(reward_max)
        
        # update algorithm
        self.update(chosen_arm = chosen_arm, reward = reward_get)
    
    simul_result = pd.DataFrame({'reward':rec_reward_get, 'max':rec_reward_max, 'correct':rec_correct})
    simul_result['regret'] = simul_result['max'] - simul_result['reward']

    return(simul_result)

class UCB1():
  def __init__(self, counts, values, alpha): # alpha는 임의로 붙임. bonus의 계수로 붙임. 
    self.counts = counts
    self.values = values
    self.alpha = alpha
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return
  
  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = self.alpha * math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return ind_max(ucb_values)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

  def run_simul(self, reward) : 
    n_observation = reward.shape[0]; n_arms = reward.shape[1]
    
    rec_reward_get = []; rec_reward_max = []; rec_correct =[]
    for i in range(n_observation) : 
        # choose action
        chosen_arm = self.select_arm()
        best_arm = reward[i].argmax()
        
        # observe and recored outcomes
        reward_get = reward[i,chosen_arm]
        reward_max = max(reward[i])
        

        if chosen_arm == best_arm :
            rec_correct.append(1)
        else :
            rec_correct.append(0)
        
        rec_reward_get.append(reward_get)
        rec_reward_max.append(reward_max)
        
        # update algorithm
        self.update(chosen_arm = chosen_arm, reward = reward_get)
    
    simul_result = pd.DataFrame({'reward':rec_reward_get, 'max':rec_reward_max, 'correct':rec_correct})
    simul_result['regret'] = simul_result['max'] - simul_result['reward']

    return(simul_result)

class LinUCB :
    def __init__(self, alpha, n_features) :
        self.alpha = alpha
        self.n_features = n_features
        return
    
    def initialize(self, n_arms) : 
        self.n_arms = n_arms
        
        self.A = np.array([np.identity(self.n_features)] * n_arms) 
        self.A_inv = np.array([np.identity(self.n_features)] * n_arms) 
        self.b = np.zeros((n_arms, self.n_features, 1))
        
    def select_arm(self,x) :         
        theta = self.A_inv @ self.b 
        UCB = np.transpose(x) @ theta + self.alpha * np.sqrt(np.transpose(x) @ self.A_inv @ x)
        
        return random.sample(list(np.where(UCB == np.max(UCB))[0]),1)[0], theta
        
    def update(self, chosen_arm, x, reward) : 
        x = x.reshape(self.n_features,-1) 
        
        # update A & A_inv
        self.A[chosen_arm] = self.A[chosen_arm] +x @ np.transpose(x)
        self.A_inv[chosen_arm] = np.linalg.inv(self.A[chosen_arm])
        
        # update b
        self.b[chosen_arm] = self.b[chosen_arm]+reward*x
        return 
    
    def run_simul(self, X, reward) : 
        n_observation = reward.shape[0]; n_arms = reward.shape[1]

        rec_reward_get = []; rec_reward_max = []; rec_correct =[]
        for i in range(n_observation) : 
            x = X[i]
            
            # choose action
            chosen_arm, _ = self.select_arm(x)
            best_arm = reward[i].argmax()
            
            # observe and recored outcomes
            reward_get = reward[i,chosen_arm]
            reward_max = max(reward[i])

            if chosen_arm == best_arm :
                rec_correct.append(1)
            else :
                rec_correct.append(0)

            rec_reward_get.append(reward_get)
            rec_reward_max.append(reward_max)

            # update algorithm
            self.update(chosen_arm = chosen_arm, x=x ,reward = reward_get)

        simul_result = pd.DataFrame({'reward':rec_reward_get, 'max':rec_reward_max, 'correct':rec_correct})
        simul_result['regret'] = simul_result['max'] - simul_result['reward']

        return(simul_result)
