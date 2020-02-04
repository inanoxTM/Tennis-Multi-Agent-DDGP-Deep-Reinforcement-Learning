import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 9e-3              # for soft update of target parameters
TARGET_SCORE = 0.5
SINGLE_AGENT = False          
EPSILON = 1.0           
EPSILON_DECAY = 1e-6    
n_episodes=5000         
max_t=10000           
LR_DECAY = True
LR_DECAY_STEP = 1
LR_DECAY_GAMMA = 0.8
UPDATE_EVERY = 40       
NUM_EPOCHS = 10  
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 0.0007       # learning rate of the actor
LR_CRITIC = 0.0007      # learning rate of the critic
LEARN_NUM = 5           # number of learning passes
TAU = 8e-3              # for soft update of target parameters
eps_start = 5   
eps_p = 300        
eps_end = 0           

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



