from classes import Agent
from numpy.random import choice

class PlayerMC(Agent):
    '''agent learning with on-policy first-visit Monte Carlo method'''
    def __init__(self, epsilon = 0.1, discount_rate = 1):
        super().__init__(values_type = 'action')
        self.epsilon = epsilon # epsilon value for epsilon-greedy policy
        self.appearances = dict() # number of appearances of state-action pairs
        self.episode_steps = [] # states-action pairs that appeared during one hand
        self.discount_rate = discount_rate # discount rate for future rewards

    def select_action(self, training_phase = True):
        '''epsilon-greedy action selection in a given state
        with respect to current state-action values'''
        # for hand values less than 12 always hit
        if self.hand.value <= 11:
            return 'hit'
        # during test phase the agent always chooses greedy action
        epsilon = self.epsilon if training_phase else 0
        # exploration vs. exploitation decision
        if choice([0,1], p = [epsilon, 1 - epsilon]) == 0:
            # random action - exploration
            return choice(['hit', 'stand'], p = [0.5, 0.5])
        else:
            # greedy action - exploitation
            if self.values[self.state]['hit'] > self.values[self.state]['stand']:
                return 'hit'
            elif self.values[self.state]['hit'] < self.values[self.state]['stand']:
                return 'stand'
            else:
                return choice(['hit', 'stand'], p = [0.5, 0.5])

    def play(self, epoch = -1, training_phase = True):
        '''simulation of one episode/hand'''
        self.episode_steps.clear()
        if training_phase:
            self.hands_played += 1
        while self.hand.value <= 21:
            action = self.select_action(training_phase)
            self.episode_steps.append((self.state, action))
            if action == 'stand':
                break
            self.deal_card()
        return self.hand.value

    def propagate_reward(self, reward):
        '''changing values of state-action pairs, reward is one of {-1,0,1}'''
        if reward == 1:
            self.hands_won += 1
        elif reward == 0:
            self.hands_drawn += 1
        # reward is available at the end of the hand, intermediate rewards = 0
        self.episode_steps.reverse()
        for state, action in self.episode_steps:
            # first -> update number of appearances for state-action pair
            if (state, action) not in self.appearances:
                self.appearances[(state, action)] = 1
            else:
                self.appearances[(state, action)] += 1
            # running average value for state-action pair expected return
            self.values[state][action] += (reward - self.values[state][action])/self.appearances[(state, action)]
            reward = self.discount_rate*reward # discounted reward