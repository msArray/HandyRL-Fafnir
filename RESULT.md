# Fafnirの強化学習の過程とparameter

## 学習開始時

config.yaml

```yaml

env_args:
    env: 'fafnir_env'

train_args:
    turn_based_training: True
    observation: False
    gamma: 0.95
    forward_steps: 32
    compress_steps: 4
    burn_in_steps: 0  # for RNNs
    entropy_regularization: 5.0e-3
    entropy_regularization_decay: 0.1
    update_episodes: 1000
    batch_size: 300
    minimum_episodes: 4000
    maximum_episodes: 200000
    epochs: -1
    num_batchers: 2
    eval_rate: 0.1
    worker:
        num_parallel: 64
    lambda: 0.7
    policy_target: 'UPGO'
    value_target: 'VTRACE'
    eval:
        opponent: ['random']
    seed: 0
    restart_epoch: 0


worker_args:
    server_address: ''
    num_parallel: 32
```

## 1000epochs程度から

config.yaml

```yaml

env_args:
    env: 'fafnir_env'

train_args:
    turn_based_training: True
    observation: False
    gamma: 0.97
    forward_steps: 32
    compress_steps: 4
    burn_in_steps: 0  # for RNNs
    entropy_regularization: 5.0e-3
    entropy_regularization_decay: 0.1
    update_episodes: 1000
    batch_size: 300
    minimum_episodes: 4000
    maximum_episodes: 200000
    epochs: -1
    num_batchers: 2
    eval_rate: 0.1
    worker:
        num_parallel: 64
    lambda: 0.7
    policy_target: 'UPGO'
    value_target: 'VTRACE'
    eval:
        opponent: ['random']
    seed: 0
    restart_epoch: 1000


worker_args:
    server_address: ''
    num_parallel: 32
```