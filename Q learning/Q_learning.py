from classes import Agent
from numpy.random import choice

class PlayerQL(Agent):
    '''agent learning with Q-learning method'''
    def __init__(self, scheduler, epsilon = 0.1, step_size = 0.5, discount_rate = 1):
        super().__init__(values_type = 'action')
        self.epsilon = epsilon # epsilon value for epsilon-greedy policy
        self.step_size = step_size # step size(learning rate), alpha parameter
        self.initial_step_size = step_size
        self.discount_rate = discount_rate # discount rate for future rewards
        self.scheduler = scheduler # scheduler for step-size during training
        self.step_size_history = [] # step-size throught the training

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
        # all intermediate rewards are 0, reward is available at the end
        if training_phase:
            self.hands_played += 1
            self.step_size = self.scheduler(epoch)
            self.step_size_history.append(self.step_size)
        while self.hand.value <= 21:
            action = self.select_action(training_phase)
            state = self.state # state before taking action
            if action == 'stand':
                break
            self.deal_card() # action = hit
            if training_phase:
                # learning while playing
                if self.hand.value > 21:
                    # in the case player goes over 21, reward is -1
                    self.values[state]['hit'] += self.step_size*(-1 - self.values[state]['hit'])
                else:
                    # value of the best action in the new state
                    max_value = max(self.values[self.state]['hit'], 
                            self.values[self.state]['stand'])
                    # learning step, update value for old state and action
                    self.values[state][action] += self.step_size*(self.discount_rate*max_value - self.values[state][action])
        return self.hand.value

    def propagate_reward(self, reward):
        if reward == 1:
            self.hands_won += 1
        elif reward == 0:
            self.hands_drawn += 1
        # in the case last action was 'stand'
        if self.hand.value <= 21:
            self.values[self.state]['stand'] += self.step_size*(reward - self.values[self.state]['stand'])