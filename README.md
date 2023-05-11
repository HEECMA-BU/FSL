# Federated Split Learning implementations

## Intro
This repo contains the three implementations for reproducibility of our FSL paper.
- Poster CoNEXT 20: Valeria Turina, Zongshun Zhang, Flavio Esposito, and Ibrahim Matta. 2020. Combining split and federated architectures for efficiency and privacy in deep learning. In Proceedings of the 16th International Conference on emerging Networking EXperiments and Technologies (CoNEXT '20). Association for Computing Machinery, New York, NY, USA, 562–563. DOI:https://doi.org/10.1145/3386367.3431678
- Paper IEEE Cloud 2021: V. Turina, Z. Zhang, F. Esposito, and I. Matta, “Federated or split? a performance and privacy analysis of hybrid split and federated learning architectures,” in IEEE CLOUD, 2021. DOI:http://dx.doi.org/10.1109/CLOUD53861.2021.00038
- Journal Paper IEEE Transaction of Bid Data is under review.

## Directories
- FSL_algorithm/models directory we have the model implementations (local emulation and distributed. Notice that the distributed implementation is only a proof of concept which lacks support for DP and the distributed communication framework PyGrid has been obsolete.)
- FSL_algorithm/resources we have some helper functions for dataloader and neural network definition.
- FSL_algorithm/attacker there is the attacker model used to reconstruct data that each client send to server.

## Requirement(Local Emulation)

It would be needed to follow these steps to obtain a correctly configured environment:

- system update
```
sudo apt-get update
```
- install venv and create a new env:
```
sudo apt-get update
sudo apt install python3-venv
python3 -m venv env
source env/bin/activate
```
- check python and pip versions
```
pip install --upgrade pip
```
### install dependencies (require python3.6/3.7/3.8 for pip) :
- syft-0.2.9 dependencies
``` 
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo apt install libsrtp2-dev	
sudo apt-get install -y libavformat-dev
sudo apt-get install libavdevice-dev
```
Next, please consider unzip the [env.zip](https://drive.google.com/file/d/1ClL5ZlRQcKeE6RmUsoaJHpY2V3DNtYr2/view?usp=sharing) at repo root to simplify the setup, especially for DP implementations, where the syft version is not accessible anymore.
```
wget -O env.zip "https://docs.google.com/uc?export=download&confirm=t&id=1ClL5ZlRQcKeE6RmUsoaJHpY2V3DNtYr2"
unzip env.zip
```
Otherwise, please follow the next steps. This setup only works with the no privacy (Privacy Oblivious) and progressive_approach (CPA-DC) implementations.
```
python3 -m pip install syft==0.2.9
python3 -m pip install opacus==0.11.0
python3 -m pip install torch==1.6.0 
pip install -U scikit-learn scipy matplotlib
``` 
- update cuda drivers to version 10.2 following https://developer.nvidia.com/cuda-10.2-download-archive
``` 
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
``` 
- add the following lines in the .bashrc and then logout if youre using ssh and login again in your venv
``` 
	# set PATH so it includes CUDA bin if it exists
	if [ -d "/usr/local/cuda-10.2/bin" ] ; then
			PATH="/usr/local/cuda-10.2/bin:$PATH"
	fi

	# set PATH so it includes CUDA bin if it exists
	if [ -d "/usr/local/cuda-10.2/lib64" ] ; then
			LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH"
	fi
``` 
- check cuda is v10.2
``` 
nvidia-smi
``` 
- download pretrained vgg16 weights for pytorch
``` 
wget -P FSL_algorithm/resources/ https://download.pytorch.org/models/vgg16-397923af.pth
``` 
- (optional) install wheel   
``` 
pip install wheel
python setup.py bdist_wheel
``` 
- (optional) install setuptools-rust
``` 
pip install setuptools-rust
``` 
 	
- update the model value (MODELS = [psl_no_privacy_vary_partition_size_fix_dataset]) in the following path
``` 
vim FSL_algorithm/resources/config.py 
``` 
- if youre using gpus then change the number of available gpus in the following ways:
	- learner 
	```
	python3 learner_script.py mnist gpu_idx
	```
	- attacker
	```
	(device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')) in attack_script.py 
	```
- (optional) run program with tmux to detach your ssh
```
tmux
tmux attach -t 0
```


## Requirement(Distributed)
Please notice that PyGrid has been obsolete. This implementation is only a proof of concept. We are working on using PyTorch RPC to rewrite the codebase.

- Install PySyft
```
python3 -m pip install syft==0.2.8
```

- clone the following repo.
```
PyGrid@4ad52e36bc13f2d15a1df395187f853526b98f1f
```
## How to Run
### Learner(Local Emulation)
```
python3 learner_script.py mnist 0 CUTS 6 new_config_key2 new_config_value2 new_config_key3 new_config_value3
```

### Learner(Distributed)
- on one node (orchestrator):
```
python3 learner_script.py mnist
```
- on each of other nodes (e.g., for FSL with one split and two pairs, we need $2*2=4$ workers)
    - (following the tutorial at https://github.com/OpenMined/PyGrid/tree/4ad52e36bc13f2d15a1df395187f853526b98f1f#manual-start)


### Attacker
please remember to specify `config.WD` and `config.data` to the experiment you want to attack. E.g., "m2_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 4]_cifar10_equDiff_SerAvg".
```
python3 attack_script.py # python3 attack_script.py cut_idx gpu_idx epoch_idx INTERMEDIATE_DATA_DIR(i.e, Train/)
```

### Experiment Configurations
All configurations i,e, DC frequency, noise_multiplier, number of clients, can be find in FSL_algorithm/resources/config.py
