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
it would be needed to follow these steps to obtain a correctly configured environment:

-system update
	$sudo apt-get update
	
-install venv and create a new env:
    	$sudo apt-get update
   	$sudo apt install python3-venv
   	$python3 -m venv env
   	$source env/bin/activate

-check python and pip versions
   	$pip install --upgrade pip

-install dependencies (require python3.6/3.7/3.8 for pip)   
	# syft-0.2.9 dependencies
	$sudo apt install libsrtp2-dev	
	$sudo apt-get install -y libavformat-dev
	$sudo apt-get install libavdevice-dev

	<!-- $pip install git+https://github.com/OpenMined/PySyft.git@syft_0.2.x -->
	$python3 -m pip install syft==0.2.9
	$python3 -m pip install opacus==0.11.0
  	$python3 -m pip install torch==1.6.0 
   	$pip install -U scikit-learn scipy matplotlib

-update cuda drivers to version 10.2 following https://developer.nvidia.com/cuda-10.2-download-archive
   	$wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   	$sudo sh cuda_10.2.89_440.33.01_linux.run

-add the following lines in the .bashrc and then logout if youre using ssh and login again in your venv
   	$vim ../.bashrc
		# set PATH so it includes CUDA bin if it exists
		if [ -d "/usr/local/cuda-10.2/bin" ] ; then
    			PATH="/usr/local/cuda-10.2/bin:$PATH"
		fi

		# set PATH so it includes CUDA bin if it exists
		if [ -d "/usr/local/cuda-10.2/lib64" ] ; then
    			LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
		fi

-check cuda is v10.2
   	$nvidia-smi

-download pretrained vgg16 weights for pytorch
	$wget -P FSL_algorithm/resources/ https://download.pytorch.org/models/vgg16-397923af.pth

-if required install wheel   
   	$pip install wheel
   	$python setup.py bdist_wheel

-if required install setuptools-rust
   	$pip install setuptools-rust

 	
-(optional) it is needed to update the model value (MODELS = [psl_no_privacy_vary_partition_size_fix_dataset]) in the following path
	$vim FSL_algorithm/resources/config.py 
 
-(optional) if youre using gpu then change the number of available gpus (device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')) in the following path
   	$vim attack_script.py 
   	
-(optional) run commands in tmux to close your ssh
   	$tmux
   	$tmux attach -t 0

- to install the pckage, in the root folder FSL run 
	$python env_setup.py develop

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
