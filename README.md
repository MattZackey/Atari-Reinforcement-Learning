## subsection{Q-Learning}
Q-learning is a mode-free RL algorithm, where we aim to approximate the optimal action value function $Q^*(s,a)$, which is the expected return for taking action $a$ in state $s$, and thereafter following the optimal policy. 

Suppose our estimate is defined as $Q(s,a)$ (also referred to as the Q-function). After taking an action and observing the next state $s'$, we calculate the Bellman error as
'''latex
$$
    \delta = r + \gamma \max_{a'}Q(s', a') - Q(s, a)
$$
where $\gamma$ is the discount factor. The Bellman error tell us the difference between: 
\begin{itemize}
    \item The current Q-value estimate $Q(s, a)$
    \item The new estimate with the immediate reward and discounted future rewards from the next state. This estimate is often referred to as our Bellman targets
\end{itemize}


## Breakout

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 13500</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="game_results/Breakout/agent_13500.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 5000</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="game_results/Breakout/agent_5000.gif" alt="Image 2" style="max-width: 70%; width: 200px;">
    </div>
  </div>

   <div style="flex: 1; text-align: center;">
    <h3>Episode 10000</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="game_results/Breakout/agent_10000.gif" alt="Image 3" style="max-width: 70%; width: 200px;">
    </div>
  </div>

</div>
  The following figure shows the total reward the agent achieved for each espisode of training.

<p align="center">
<img src="game_results/Breakout/episode_scores_13500.png" width="800"/>
</p>


## Future Work

## References

## Citation

If you find this code helpful, kindly reference the following:

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
