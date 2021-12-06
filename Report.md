# Learning Algorithm

Prior to advent of deep neural networks, value-based models use Q-tables in order to induce optimal policies. However, this approach has serious limitations since the size of Q-tables become too large to process when dealing with huge state spaces. Deep neural networks come into play to rescue this problem. Deep neural networks allow for approximation of Q-values even in very large state spaces. The value-based algorithms using deep neural networks to approximate Q-values are called Deep Q networks (DQN). 

### Building a deep neural network (see the file `model.py`)
In order to implement DQN in practice, you first need to model a deep neural network for estimating Q-values. The file `model.py` shows the whole process of building a deep neural network. The file `model.py` employs Pytorch’s typical approach of building a neural network. Thus if you are familiar with Pytorch, the whole process will make sense immediately. 

### Replay buffer (see the second `class` of `dqn_agent.py`)
A notable feature in DQN is a (experience) replay buffer. This replay buffer acts as a memory that serves two purposes. 
1.	Saving the agent’s experience: an interaction between the agent and the environment provides transition information in the form of tuple (s, a, r, s’), where s is the current state, a is the action, r is the reward, and s’ the next state. This transition information is basically the agent's experience. The replay buffer stores the agent's experience over several episodes.
2.	Sampling the agent’s experiences at random: using a **randomly** sampled data from the replay buffer is very useful when we train our DQN agent. I emphasized the word ‘randomly’ as this is very crucial. Without the replay buffer, agent would have been trained based on sequential experiences where the current experience is always followed by the next experience. However, these sequential experiences will be highly correlated as situation in the current state s and situation in the next state s’ are highly correlated. This highly correlated experiences will make our neural network easily overfit. However, random selection of the agent’s experience from the replay buffer reduces the correlation between experiences. 
This process is described in the `class` `ReplayBuffer` (see the second part of `dqn_agent.py’`).  

### DQN agent (see the first `class` of `dqn_agent.py`)
Here's a big picture of how the agent and the environment interact with each other. The environment provides a state to the agent. The agent needs to act in response to the state provided by the environment. This process is contained in the function `act()`. The function describes the process of how the agent select actions in response to the state provided by the environment. Once the agent select an action, the environment will react to it (see `env.step(action)` in the file `Navigation.ipynb`) and return the reward and the next state to the agent. Thus, agent's action and environment's reaction produce experience tuple (state, action, reward, next state). The function `agent.step()` saves this experience in the `ReplayBuffer` and also randomly select experiences from the `ReplayBuffer`. Moreover, the function `agent.step()` updates the parameters of local and target networks through the function `learn()`. The function `learn()` uses backward autograd calculation to obtain derivatives of loss function. The derivatives are used to update the parameters of local and target networks.

# Plots of Rewards

<p align="center">
<img width="50%" src="https://user-images.githubusercontent.com/95396618/144888270-2d480375-c245-40e5-89b0-344d78463835.PNG"/>  
</p> 


<p align="center">
<img width="70%" src="https://user-images.githubusercontent.com/95396618/144886441-e0bda08b-8ea2-4b4d-90fd-be578016170c.PNG"/>  
</p>  



# Ideas for Future Work

### Batch normalization or Droupout
To make the learning algorithm more efficient, we can add bath normalization layer or dropout layer to the file `model.py`.

### Double Q-Learning
Since Deep Q-Learning tends to overestimate action values, it is worthwhile to try Double Q-Learning

### Prioritized experienced replay 
The agent might learn more effectively from some transitions than from others. Thus, it would produce better performance if the more important experience tuples are being sampled with higher probability. Rather than uniformly sampling experience tuples from the replay buffer, one can prioritize experiences that give the agent more chance to learn more effectively.
