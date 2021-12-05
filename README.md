# Project1: Navigation

### The Environment
For this project, you will work with the Unity Banana environment.

<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/95396618/144585373-58159d73-e732-4647-9034-37c00778b9b1.png"/>  
</p>  
<p align="center">
Unity ML-Agents Tennis Environment 
</p>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single **score** for each episode.  

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Download the Unity Environment

For this project, you will **not** need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

* Mac OSX: <a href="https://github.com/AIinFinance/Project3-Collaboration-and-Competition/files/7648933/Tennis.app.zip" download>click here</a>
* Windows (32-bit): <a href="https://github.com/AIinFinance/Project3-Collaboration-and-Competition/files/7648930/Tennis_Windows_x86.zip" download>click here</a>
* Windows (64-bit): <a href="https://github.com/AIinFinance/Project3-Collaboration-and-Competition/files/7648886/Tennis_Windows_x86_64.zip" download>click here</a>

Then, place the file in your folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

### Instructions

To start training the agent, open Tennis.ipynb on Jupyter Notebook and to run the code cell use Shift+Enter or click the Run button.
