# Federated Split Learning implementations

## Intro
This repo contains the three implementations for reproducibility of our FSL paper.
Poster CoNEXT 20: https://conext20posters.hotcrp.com/doc/conext20posters-paper83.pdf?cap=083aTcMf9NZevb4
Paper IEEE Cloud 2021 is coming.

## Directories
    - FSL_algorithm/models directory we have the model implementations (simulation and remote, the remote folder is in progress)
    - FSL_algorithm/resources we have some helper functions for dataloader and neural network definition.
    - FSL_algorithm/attacker there is the attacker model used to reconstruct data that each client send to server.


## Requirement
```
- in the folder FSL_algorithm run "python setup.py develop" to nstall the package
Then install the following packages:
    - syft==0.2.9
    - torch==1.6.0    # require python3.6/3.7/3.8 for pip
```

## How to Run
### Learner
python3 learner_script.py mnist

### Attacker
python3 attack_script.py

### experiment Configurations
All configurations i,e, DC frequency, noise_multiplier, number of clients, can be find in FSL_algorithm/resources/config.py, 


