#


## DrQv2
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drqv2.py/#L42)
```python 
DrQv2(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 2000,
   storage_size: int = 1000000, feature_dim: int = 50, batch_size: int = 256,
   lr: float = 0.0001, eps: float = 1e-08, hidden_dim: int = 1024,
   critic_target_tau: float = 0.01, update_every_steps: int = 2,
   stddev_clip: float = 0.3, init_fn: str = 'orthogonal'
)
```


---
Data Regularized Q-v2 (DrQv2) agent.
Based on: https://github.com/facebookresearch/drqv2


**Args**

* **env** (VecEnv) : Vectorized environments for training.
* **eval_env** (VecEnv) : Vectorized environments for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on the pre-training mode.
* **num_init_steps** (int) : Number of initial exploration steps.
* **storage_size** (int) : The capacity of the storage.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **batch_size** (int) : Number of samples per batch to load.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **critic_target_tau**  : The critic Q-function soft-update rate.
* **update_every_steps** (int) : The agent update frequency.
* **stddev_clip** (float) : The exploration std clip range.
* **init_fn** (str) : Parameters initialization method.



**Returns**

DrQv2 agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drqv2.py/#L147)
```python
.update()
```

---
Update the agent and return training metrics such as actor loss, critic_loss, etc.

### .update_critic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drqv2.py/#L177)
```python
.update_critic(
   obs: th.Tensor, actions: th.Tensor, rewards: th.Tensor, discount: th.Tensor,
   next_obs: th.Tensor
)
```

---
Update the critic network.


**Args**

* **obs** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **discounts** (th.Tensor) : discounts.
* **next_obs** (th.Tensor) : Next observations.


**Returns**

None.

### .update_actor
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/drqv2.py/#L224)
```python
.update_actor(
   obs: th.Tensor
)
```

---
Update the actor network.


**Args**

* **obs** (th.Tensor) : Observations.


**Returns**

None.
