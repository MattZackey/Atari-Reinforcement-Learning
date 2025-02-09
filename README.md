# Deep Q-Learning 🤖 

## Overview
This repository implements the Deep Q-Network (DQN) algorithm within Atari 2600. Currently, the results for Breakout are available, but additional games will be added soon. The project aims to replicate some of the success achieved by DeepMind in their paper, [Human-level control through deep reinforcement learning](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf). In their work, DeepMind demonstrated that a DQN agent can learn successful policies directly from raw pixel inputs, achieving human-level performance on a variety of Atari games. This repository serves as an exploration of this breakthrough.

## Breakout 

The trained DQN agent achieves a total score of 343. Although, this is not a perfect score, it still highlights a high-level of competence in the game. Notably, the agent has learned that creating a tunnel along both sides is the most efficient strategy for breaking as many bricks as possible. With this strategy, the agent decreases the number of times it must hit the ball, while maximising the number of collisions. Similarly, DeepMind found the same relationship in their training. Highlighting, that such a strategy emerges naturally in a sufficiently trained agent. <br>
<h3 align="center"><b>Results</b></h3>

<p align="center">
  <img src="results/BreakoutDeterministic-v4/gameplay/agent_13500.gif" width="400">
</p>
<br>
The total reward the agent achieved for each espisode of training.
<br>
<p align="center">
<img src="results/BreakoutDeterministic-v4/plots/episode_scores_13500.png" width="800"/>
</p>
<br>

## Algorithm Details

### Q-Learning
Q-learning is a mode-free RL algorithm, where we aim to approximate the optimal action value function $Q^*(s,a)$, which is the expected return for taking action $a$ in state $s$, and thereafter following the optimal policy. 

Suppose our estimate is defined as $Q(s,a)$ (also referred to as the Q-function). After taking an action and observing the next state $s′$, we calculate the Bellman error as

$$\delta = r + \gamma \max_{a′}Q(s′, a′) - Q(s, a)$$

where $\gamma$ is the discount factor. The Bellman error tell us the difference between: 
* The current Q-value estimate $Q(s, a)$
* The new estimate with the immediate reward $r$ and discounted future rewards from the next state $\gamma \max_{a′}Q(s′, a′)$. This estimate is often referred to as our Bellman targets

We update our Q-function as

$$Q(s,a) \leftarrow Q(s,a) + \alpha \delta$$

where $\alpha$ is the learning rate.

### Deep Q-learning

In Q-learning, the action-value function must be estimated across all states and actions. However, in many environments, the state space is too large to store a Q-value for every state action pair. As a result, a function approximator is used 

$$Q(s,a, \textbf{w}) \approx Q(s,a),$$

where $\textbf{w}$ are the parameters. This allows us to generalise from seen to unseen states. In Deep Q-learning, a neural network is used as the function approximator, and it is typical to refer to it as the Q-network.  

### Experience Replay

To improve training stability, Deep Q-learning uses an experience replay buffer. The agent stores its experiences in a buffer, which consists of tuples of the form $(s, a, r, s′)$. During training a random batch is sampled from the replay buffer, and the parameters of the Q-network are updated. By selecting random samples from our replay buffer, we break the correlation between consecutive samples, stabilizing training.

### Fixed Q-targets

To further stabilize the learning process, Deep Q-Learning makes use of a target network, which is a copy of the Q-network. The target network is responsible for calculating the Bellman targets

$$y = r + \max_{a′}Q(s′, a′; \mathbf{w}^-)$$

where $w^-$ denote the parameters of the target network. These parameters are only updated periodically as copies of the Q-networks parameters.

### Off-Policy
Deep Q-learning is a off-policy RL algorithm, where we find a target policy (the policy that we ultimately want to optimize) using data generated by a different policy, called the behavior policy. The behavior policy is an exploratory policy, whereas target policy is a greedy policy.

The behavior policy used in Q-learning is a epsilon-greedy strategy:
* With probability $\epsilon$, choose a random action (exploration)
* With probability $1 - \epsilon$, choose the action that maximizes the Q-function (exploitation).

During the training process, $\epsilon$ decays to some point, resulting in the agent exploring less and exploiting more as it learns.

### Training
The Q-network is trained by minimizing the Mean Squared Bellman Error (MSBE)

$$E_{\{s, a, r, s′\} \sim P}\left[\left(r + \gamma \max_{a′}Q(s', a′; \mathbf{w}^-) - Q(s, a; \mathbf{w})\right)^2\right]$$

where $P$ denotes our environment. This expectation unfortunately cannot be computed, because the dynamics of the environment are unknown. As a result, after each step in the environment we take a random sample from the replay buffer to approximate this expectation, and perform one step of gradient descent. 

## Atari and Agent Details

### States
In the environment setup, a state is represented by four consecutive frames.  This allows the agent to determine the motion dynamics of the environment.

### Circular Replay Buffer 
The fully trained DQN agent is just under 7 GB of memory, which is achieved by saving individual frames rather than states. These frames are stored in what is known as a circular replay buffer, which is a data structure that continuously overwrites the oldest entries. This circular replay buffer has been reproduced from scratch.

How it works:
 * Circular Replay Buffer Structure: The replay buffer consists of a pointer that indicates where the latest frame should be inserted. When the buffer is full, the pointer loops back to the start, replacing the oldest frame with the new one.
 * Stored Information: For every saved frame, the corresponding action taken, reward received, and a done flag (indicating whether the game has ended) are also stored. This ensures that each frame is accompanied by all the necessary data to reconstruct a full experience.
 * Creating the transition: The transitions tuples $$(s, a, r, s’)$$ are created when sampling from the replay buffer. That is, we sample a random batch of indices from the buffer. Suppose one of those indices is i, which corresponds to selecting frame fi, action ai and reward ri. The transition is built:
      * $$s = [f_{i-4}, f_{i-3}, f_{i-2}, f_{i-1}]$$
      * $$a = a_i$$
      * $$r = r_i$$
      * $$s′ = [f_{i-3}, f_{i-2}, f_{i-1}, f_i]$$
 * Initial Population of the Buffer: The replay buffer is designed in such a way that it must be full before sampling from it. Therefore, before training begins, the replay buffer is populated with experience from an agent taking random actions.


## Setup Instructions
So it's possible to achieve the same performance as DeepMind in Atari! If you want, you can also try:

  1. git clone `https://github.com/MattZackey/Atari-Reinforcement-Learning.git`
  2. Open Anaconda/Miniconda and navigate to `path_to_project`
  4. Create a Conda environment `conda create --name atari-rl python=3.10` and active environment `conda activate atari-rl`
  5. Install required dependencies `pip install -r requirements.txt`
  6. For Windows user you are also required to install `pip install gym[atari]`
  7. Run the training script `python train.py --game BreakoutDeterministic-v4`

## References

* Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533.

* Gordić, A. (2021). *pytorch-learn-reinforcement-learning* [GitHub repository]. GitHub. https://github.com/gordicaleksa/pytorch-learn-reinforcement-learning

## Citation

If you find this code helpful, kindly reference:

```
@misc{Zackey2025AtariReinforcementLearning,
  author = {Zackey, Matthew},
  title = {Atari-Reinforcement-Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/MattZackey/Atari-Reinforcement-Learning}},
}
```
