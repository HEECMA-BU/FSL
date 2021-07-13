# Federated Split Learning implementations

## Intro
This repo contains the three implementations for reproducibility of our FSL paper.
- Poster CoNEXT 20: Valeria Turina, Zongshun Zhang, Flavio Esposito, and Ibrahim Matta. 2020. Combining split and federated architectures for efficiency and privacy in deep learning. In Proceedings of the 16th International Conference on emerging Networking EXperiments and Technologies (CoNEXT '20). Association for Computing Machinery, New York, NY, USA, 562–563. DOI:https://doi.org/10.1145/3386367.3431678
- Paper IEEE Cloud 2021 is coming.

## Directories
    - FSL_algorithm/models directory we have the model implementations (simulation and remote, the remote folder is in progress)
    - FSL_algorithm/resources we have some helper functions for dataloader and neural network definition.
    - FSL_algorithm/attacker there is the attacker model used to reconstruct data that each client send to server.

## Requirement(Simulation)
```
- in the root folder FSL run "python env_setup.py develop" to install the package
Then install the following packages:
    - syft==0.2.9
    - torch==1.6.0    # require python3.6/3.7/3.8 for pip
```
## Requirement(Distributed)
```
- in the root folder FSL run "python env_setup.py develop" to install the package
Then install the following packages:
    - syft==0.2.8
    - PyGrid@4ad52e36bc13f2d15a1df395187f853526b98f1f
```
## How to Run
### Learner(Simulation)
```
python3 learner_script.py mnist
```

### Learner(Simulation)
```
- on one node (orchestrator):
    - python3 learner_script.py mnist
- on each of other nodes (e.g., for FSL with one split and two pairs, we need $2*2=4$ workers)
    - (following the tutorial at https://github.com/OpenMined/PyGrid/tree/4ad52e36bc13f2d15a1df395187f853526b98f1f#manual-start)
```

### Attacker
python3 attack_script.py

### experiment Configurations
All configurations i,e, DC frequency, noise_multiplier, number of clients, can be find in FSL_algorithm/resources/config.py
