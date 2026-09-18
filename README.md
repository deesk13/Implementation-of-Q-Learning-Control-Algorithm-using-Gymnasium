# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function through repeated interaction with the environment and determine a policy that allows it to reach the goal while avoiding holes.


## Software Requirements
1. Python 3.x
2. Gymnasium
3. NumPy
4. Matplotlib
5. Jupyter Notebook / Google Colab / VS Code


## Environment Description
FrozenLake-v1 is a grid-world reinforcement learning environment provided by Gymnasium. The environment consists of a 4 × 4 grid with 16 states and 4 possible actions:

Action	Meaning
0	      Left
1	      Down
2	      Right
3	      Up


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---
## Algorithm
        )

        state = new_state
        total_episode_reward += reward

    epsilon = max(
        min_epsilon,
        epsilon - epsilon_decay_rate
    )

    episode_rewards.append(total_episode_reward)




## Output

## Final Q-table:

<img width="217" height="283" alt="image" src="https://github.com/user-attachments/assets/a88d500e-100a-41f2-a26f-00e9cf0b3011" />




## Estimated State-Value Function:

<img width="231" height="92" alt="image" src="https://github.com/user-attachments/assets/9869c050-721d-4b10-ae83-19789e2c0aca" />





## Learned Policy:
<img width="147" height="92" alt="image" src="https://github.com/user-attachments/assets/660fef3e-e078-40e4-ab37-ef3ca96500b7" />




## Average reward over last 1000 episodes: 
<img width="332" height="25" alt="image" src="https://github.com/user-attachments/assets/93d6cdec-f32a-474e-9622-5bd2ddfa3529" />



## Result

The Q-Learning control algorithm was successfully implemented
using the Gymnasium FrozenLake-v1 environment.

The agent learned an action-value function through repeated
interaction with the environment and obtained a learned policy
for selecting actions. The learned Q-table, state-value function,
policy, and average reward were successfully obtained.


## Inference
Q-Learning successfully learns an optimal policy by updating the
Q-table based on the rewards obtained from the environment.

The epsilon-greedy strategy allows the agent to explore different
actions initially and gradually exploit the learned Q-values as
epsilon decreases.

After sufficient training, the agent learns suitable actions for
moving from the starting state toward the goal while avoiding
holes. The learning curve indicates the improvement in the agent's
performance during training.


