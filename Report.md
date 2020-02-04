# Report of the project


### Table of Contents

1. [Introduction](#introduction)
2. [Learning Algorithm](#LA)
3. [Hyperparameters](#hyper)
4. [Results](#Results)
5. [Next Steps](#NextSteps)


## Introduction <a name="introduction"></a>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to moves toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved when the average (over 100 episodes) of those scores is at least +0.5.

In our case, option 2 was performed, using 20 agents, as it trained faster. 


## Learning Algorithm <a name="LA"></a>

The algorithm I used for this project is the Multi-Agent DDPG Actor Critic Model ([paper](https://deepmind.com/research/publications/continuous-control-deep-reinforcement-learning)).

In order to explain this algorithm, first, it needs to know that there are two ways for estimating expected returns. First is the Monte Carlo estimate, which roles out an episode in calculating the discounter total reward from the rewards sequence. In Dynamic Programming, the Markov Decision Process (MDP) is solved by using value iteration and policy iteration. Both techniques require transition and reward probabilities to find the optimal policy. When the transition and reward probabilities are unknown, we use the Monte Carlo method to solve MDP. The Monte Carlo method requires only sample sequences of states, actions, and rewards. Monte Carlo methods are applied only to the episodic tasks.

We can approach the Monte — Carlo estimate by considering that the Agent play in episode A. We start in state St and take action At. Based on the process the Agent transits to state St+1. From environment, the Agent receives the reward Rt+1. This process can be continued until the Agent reaches the end of the episode. The Agent can take part also in other episodes like B, C, and D. Some of those episodes will have trajectories that go through the same states, which influences that the value function is computed as average of estimates. Estimates for a state can vary across episodes so the Monte Carlo estimates will have high variance.

Also, we can apply the Temporal Difference estimate. TD approximates the current estimate based on the previously learned estimate, which is also called bootstrapping. TD error are the difference between the actual reward and the expected reward multiplied by the learning raw. TD estimates are low variance because you’re only compounding a single time step of randomness instead of a full rollout like in Monte Carlo estimate. However, due to applying a bootstrapping (dynamic programming) the next state is only estimated. Estimated values introduce bias into our calculations. The agent will learn faster, but the converging problems can occur.

Deriving the Actor-Critic concept requires to consider first the policy-based approach (AGENT). As we discussed before the Agent playing the game increases the probability of actions that lead to a win, and decrease the probability of actions that lead to losses. However, such process is cumbersome due to lot of data to approach the optimal policy.

It can evaluate the value-based approach (CRITIC), where the guesses are performed on-the-fly, throughout all the episode. At the beginning our guesses will be misaligned. But over time, when we capture more experience, we will be able to make solid guesses. 

Based on this short analysis we can summarize that the Agent using policy-based approach is learning to act (agent learns by interacting with environment and adjusts the probabilities of good and bad actions, while in a value-based approach, the agent is learning to estimate states and actions.) . In parallel we use a Critic, which is to be able to evaluate the quality of actions more quickly (proper action or not) and speed up learning. Actor-critic method is more stable than value-based agents.

As a result of merge Actor-Critic we utilize two separate neural networks. The role of the Actor network is to determine the best actions (from probability distribution) in the state by tuning the parameter θ (weights). The Critic by computing the temporal difference error TD (estimating expected returns), evaluates the action generated by the Actor.

![algorithm](/images/ddgp.jpg)

## Hyperparameters <a name="hyper"></a>
The Parameters used for the Agent are:

| Hyperparameter  | value |
| ------------- | ------------- |
| Replay Buffer Size  | 1e6  |
| Minibatch Size  | 128 |
| Discount Rate  | 0.99  |
| TAU  | 9e-3  |
| Actor Learning Rate  | 0.0007  |
| Critic Learning Rate  | 0.0007  |
| Neurons Actor Netork Layer 1 | 128  |
| Neurons Actor Netork Layer 2 | 64  |
| Neurons Critic Netork Layer 1 | 128  |
| Neurons Critic Netork Layer 2 | 64  |

## Results DDGP <a name="Results"></a>
Two tests have been carried out to analyse how far the model could go. The first one (Approach 1) has been carried out to solve the objectives proposed by the exercise: To reach 0.5 of reward in 100 consecutive episodes. The models are saved when the result is reached and it has been solved in 600 episodes.
![Results](/images/approach1.png)

The next test (approach 2) as to leave the agents learning until they achieved an average score of 2 for 100 consecutive episodes. It has reached this result in 1489 episodes. The models has been saved with {app2} sufix. 

![Results](/images/approach2.png)

## Next Steps <a name="NextSteps"></a>
The next steps to improve the results can be the following: 

1) Optimize the hyperparameters of the process and network
2) Change ANN by LSTM 
3) Test other algorithm like as A3C, TD3, PPO.
4) Add Prioritized Replay
5) Add batch Normalization


