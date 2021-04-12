# Federated Split Learning implementations

## Intro
This repo contains the three implementations for reproducibility of our FSL paper. Under FSL_algorithm/models directory we have the model implemnetations. And under FSL_algorithm/resources we have some helper functions for dataloader and neural network definition.

## Requirement
```
syft==0.2.9
sklearn
opacus
torch==1.6.0
```

## How to Run
### Learner
python3 learner_script.py mnist
### Attacker
python3 attack_script.py

### experiment Configurations
All configurations i,e, DC frequency, noise_multiplier, number of clients, can be find in FSL_algorithm/resources/config.py, 


