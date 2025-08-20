#


## SAC
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/sac.py/#L41)
```python 
SAC(
   env: VecEnv, eval_env: Optional[VecEnv] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 5000,
   storage_size: int = 10000000, feature_dim: int = 50, batch_size: int = 1024,
   lr: float = 0.0001, eps: float = 1e-08, hidden_dim: int = 1024,
   actor_update_freq: int = 1, critic_target_tau: float = 0.005,
   critic_target_update_freq: int = 2, log_std_range: Tuple[float, ...] = (-5.0, 2),
   betas: Tuple[float, float] = (0.9, 0.999), temperature: float = 0.1,
   fixed_temperature: bool = False, discount: float = 0.99, init_fn: str = 'orthogonal'
)
```


---
Soft Actor-Critic (SAC) agent.
Based on: https://github.com/denisyarats/pytorch_sac


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
* **actor_update_freq** (int) : The actor update frequency (in steps).
* **critic_target_tau** (float) : The critic Q-function soft-update rate.
* **critic_target_update_freq** (int) : The critic Q-function soft-update frequency (in steps).
* **log_std_range** (Tuple[float]) : Range of std for sampling actions.
* **betas** (Tuple[float]) : Coefficients used for computing running averages of gradient and its square.
* **temperature** (float) : Initial temperature coefficient.
* **fixed_temperature** (bool) : Fixed temperature or not.
* **discount** (float) : Discount factor.
* **init_fn** (str) : Parameters initialization method.



**Returns**

PPO agent instance.


**Methods:**


### .alpha
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/sac.py/#L162)
```python
.alpha()
```

---
Get the temperature coefficient.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/sac.py/#L166)
```python
.update()
```

---
Update the agent and return training metrics such as actor loss, critic_loss, etc.

### .update_critic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/sac.py/#L194)
```python
.update_critic(
   obs: th.Tensor, actions: th.Tensor, rewards: th.Tensor, terminateds: th.Tensor,
   truncateds: th.Tensor, next_obs: th.Tensor
)
```

---
Update the critic network.


**Args**

* **obs** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Terminateds.
* **truncateds** (th.Tensor) : Truncateds.
* **next_obs** (th.Tensor) : Next observations.


**Returns**

None.

### .update_actor_and_alpha
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/sac.py/#L244)
```python
.update_actor_and_alpha(
   obs: th.Tensor
)
```

---
Update the actor network and temperature.


**Args**

* **obs** (th.Tensor) : Observations.


**Returns**

None.
