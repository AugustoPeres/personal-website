+++
title = "Moon Lander deep Q-Learning"
author = ["Augusto"]
description = """A simple implementation deep Q-Learning with experience replay
for the moon lander environment."""
draft = false
tags = ["python", "machine learning", "pytorch", "reinforcement-learning"]
showFullcontent = false
readingTime = true
hideComments = false
mathJaxSupport = true
+++


# Introduction

This is my simple implementation of deep Q-Learning for the moon-lander
environment. The code is available on my
[github](https://github.com/AugustoPeres/RL-experiments/blob/main/qlearning/README.md#moon-lander). Feel
free to reproduce the results here or even use it for your own environments:


{{< figure src="/ox-hugo/moon_lander_episode_2000.gif" caption="<span class=\"figure-number\">Figure 1: </span>Agent successfully landing the ship after 2000 episodes.">}}


# The agent

The agent receives a pytorch neural network and all the other parameters
required for training such as, the sizes of the batch to be used for training
the network, the discount for future rewards \\(\gamma\\), the starting
exploration rate \\(\epsilon\\), the minimum exploration rate and the speed of
its decay. It also receives the learning rate for the agent's neural network,
the number of actions, the target network update frequency, _i.e_, how often we
copy the parameters of agent's policy network to the target network. Finally,
the agent receives an integer that controls how many transitions we store in the
replay buffer.

```python
class QLearningAgent():

    def __init__(self,
                 network,
                 batch_size,
                 gamma,
                 start_epsilon,
                 min_epsilon,
                 decay,
                 learning_rate,
                 number_of_actions,
                 target_network_update_frequency,
                 max_steps_in_buffer=int(1e4)):

```

The agent follows a standard \\(\epsilon\\)-greedy policy:

```python
class QLearningAgent():
    # ... snip ... #
    def policy(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            return random.sample(range(self.number_of_actions), 1)[0]
        with torch.no_grad():
            q_values = self.nn(
                torch.tensor(observation).unsqueeze(0)).squeeze(-1)
        action = torch.argmax(q_values)
        return action.numpy()
```

And we use an a standard exponential \\(\epsilon\\) decay:

```python
class QLearningAgent():
    # ... snip ... #
    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
```

We update the agents weights using:

<div class="latex">

\begin{align}
Q(s_t, a_t) = r_t + \gamma * \max_{a \in Actions} Q(s_{t+1}, a)
\end{align}

</div>

when the \\(s_{t + 1}\\) is not final. When \\(s_{t+1}\\) is final we simply
use:

<div class="latex">

\begin{align}
Q(s_t, a_t) = r_t
\end{align}

</div>

Instead of having one neural network for each action our neural network instead
has \\(4\\) outputs corresponding to the q-values of each action. As such,
during training we need to mask the q-values for all actions except the one that
was actually taken for \\(s_t\\):


```python
class QLearningAgent():
    # ... snip ... #
    def update(self):
        if len(self.experience_replay) < self.batch_size:
            return
        training_batch = self.get_random_batch_from_replay_buffer()
        training_inputs = self.make_batched_input(training_batch)
        training_targets, masks = self.make_batched_target(training_batch)

        pred_q_values = self.nn(training_inputs)
        loss = torch.nn.MSELoss()(pred_q_values * masks, training_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_batched_input(self, training_batch):
        states = [torch.tensor(x[0]) for x in training_batch]
        return torch.stack(states)

    def make_batched_target(self, training_batch):
        targets = []
        masks = []
        for (observation, action, reward, next_observation,
             is_terminal) in training_batch:
            mask = torch.eye(self.number_of_actions)[action]
            if is_terminal:
                targets.append(mask * reward)
            else:
                with torch.no_grad():
                    preds = self.target_network(
                        torch.tensor(next_observation).unsqueeze(0))
                max_q_value = torch.max(preds)
                targets.append(mask * (reward + self.gamma * max_q_value))
            masks.append(mask)
        return torch.stack(targets), torch.stack(masks)
```

# The training loop

The training loop consists on the steps:

1. The agent sends an action to the environment;
2. The agent collects a reward and an observation from the environment;
3. We update the agent;

Or, in python code:

```python
def train_agent(number_of_episodes, agent, record_evey_n_episodes=100):
    env = gym.make('LunarLander-v2')
    for e in range(number_of_episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            action = agent.policy(observation)
            new_observation, reward, terminated, truncated, _ = env.step(
                action)

            agent.collect_observation(observation, action, reward,
                                      new_observation, terminated)
            agent.update()
            agent.update_target_network()
            agent.epsilon_decay()

            observation = new_observation.copy()

```
