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
       Algorithm — Q-Learning Control using FrozenLake-v1
Step 1: Initialize the environment

Create the FrozenLake-v1 environment with a 4 × 4 grid.

Step 2: Initialize the Q-table

Create a Q-table with:

16 states
4 actions

Initialize all Q-values to zero.

Step 3: Set hyperparameters

Set:

Learning rate α = 0.1
Discount factor γ = 0.99
Initial exploration rate ε = 1.0
Minimum exploration rate εmin = 0.01
Epsilon decay rate = 0.0001
Number of training episodes = 10000
Step 4: Select an action using epsilon-greedy

For the current state:

Generate a random number.
If the number is less than ε, select a random action.
Otherwise, select the action having the maximum Q-value.
Step 5: Perform the action

Execute the selected action in the environment and obtain:

Next state
Reward
Termination status
Truncation status
Step 6: Calculate the target

For a non-terminal state:

$$ Target = R_{t+1}+\gamma\max_a Q(S_{t+1},a) $$

For a terminal state:

$$ Target = R_{t+1} $$
Step 7: Update the Q-value
$$ Q(S_t,A_t)\leftarrow Q(S_t,A_t)+ \alpha[Target-Q(S_t,A_t)] $$
Step 8: Move to the next state

Set:

state = next_state

Continue until the episode terminates or is truncated.

Step 9: Decay epsilon

After every episode:

$$ \epsilon=\max(\epsilon_{min},\epsilon-\text{decay}) $$

This gradually changes the agent from exploration to exploitation.
## PROGRAM
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Create FrozenLake environment
# --------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

# Number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

print("Number of states :", num_states)
print("Number of actions:", num_actions)


# --------------------------------------------------
# 2. Initialize Q-table
# --------------------------------------------------

Q = np.zeros((num_states, num_actions))


# --------------------------------------------------
# 3. Hyperparameters
# --------------------------------------------------

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
min_epsilon = 0.01   # Minimum exploration rate
epsilon_decay_rate = 0.0001

num_episodes = 10000


# --------------------------------------------------
# 4. Store rewards
# --------------------------------------------------

episode_rewards = []


# --------------------------------------------------
# 5. Q-Learning training
# --------------------------------------------------

for episode in range(num_episodes):

    state, info = env.reset()

    done = False
    total_episode_reward = 0

    while not done:

        # ------------------------------------------
        # Epsilon-greedy action selection
        # ------------------------------------------

        if np.random.random() < epsilon:
            # Exploration
            action = env.action_space.sample()

        else:
            # Exploitation
            action = np.argmax(Q[state, :])

        # ------------------------------------------
        # Take action
        # ------------------------------------------

        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # ------------------------------------------
        # Q-Learning update
        # ------------------------------------------

        if terminated:
            # No future value after terminal state
            target = reward

        else:
            # Bellman optimality target
            target = reward + gamma * np.max(Q[new_state, :])

        Q[state, action] = Q[state, action] + \
            alpha * (target - Q[state, action])

        # ------------------------------------------
        # Move to next state
        # ------------------------------------------

        state = new_state

        total_episode_reward += reward

    # ----------------------------------------------
    # Decay epsilon
    # ----------------------------------------------

    epsilon = max(
        min_epsilon,
        epsilon - epsilon_decay_rate
    )

    episode_rewards.append(total_episode_reward)


# --------------------------------------------------
# 6. Display final Q-table
# --------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))


# --------------------------------------------------
# 7. Calculate state-value function
# --------------------------------------------------

state_values = np.max(Q, axis=1)

print("\nEstimated State-Value Function:")
print(np.round(state_values, 3))


# --------------------------------------------------
# 8. Extract learned policy
# --------------------------------------------------

policy = np.argmax(Q, axis=1)

print("\nLearned Policy:")
print(policy)


# --------------------------------------------------
# 9. Calculate average reward
# --------------------------------------------------

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:",
      average_reward)


# --------------------------------------------------
# 10. Plot learning curve
# --------------------------------------------------

window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Performance on FrozenLake")

plt.grid(True)
plt.show()


# --------------------------------------------------
# 11. Close environment
# --------------------------------------------------

env.close()
```




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


