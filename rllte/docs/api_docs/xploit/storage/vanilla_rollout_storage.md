#


## VanillaRolloutStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L36)
```python 
VanillaRolloutStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 256, batch_size: int = 64, num_envs: int = 8,
   discount: float = 0.999, gae_lambda: float = 0.95
)
```


---
Vanilla rollout storage for on-policy algorithms.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device to convert the data.
* **storage_size** (int) : The capacity of the storage. Here it refers to the length of per rollout.
* **batch_size** (int) : Batch size of samples.
* **num_envs** (int) : The number of parallel environments.
* **discount** (float) : The discount factor.
* **gae_lambda** (float) : Weighting coefficient for generalized advantage estimation (GAE).


**Returns**

Vanilla rollout storage.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L69)
```python
.reset()
```

---
Reset the storage.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L89)
```python
.add(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, infos: Dict,
   next_observations: th.Tensor, log_probs: th.Tensor, values: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **observations** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Termination signals.
* **truncateds** (th.Tensor) : Truncation signals.
* **infos** (Dict) : Extra information.
* **next_observations** (th.Tensor) : Next observations.
* **log_probs** (th.Tensor) : Log of the probability evaluated at `actions`.
* **values** (th.Tensor) : Estimated values.


**Returns**

None.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L129)
```python
.update()
```

---
Update the terminal state of each env.

### .compute_returns_and_advantages
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L134)
```python
.compute_returns_and_advantages(
   last_values: th.Tensor
)
```

---
Perform generalized advantage estimation (GAE).


**Args**

* **last_values** (th.Tensor) : Estimated values of the last step.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_rollout_storage.py/#L159)
```python
.sample()
```

---
Sample data from storage.
