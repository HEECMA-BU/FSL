import os
import json

# simulation implementation
import FSL_algorithm.models.simulation.no_privacy.model1_no_privacy_fix_partition_size_vary_dataset as psl_no_privacy_fix_partition_size_vary_dataset
import FSL_algorithm.models.simulation.no_privacy.model2_no_privacy_fix_partition_size_vary_dataset as fsl_no_privacy_fix_partition_size_vary_dataset
import FSL_algorithm.models.simulation.no_privacy.model1_no_privacy_vary_partition_size_fix_dataset as psl_no_privacy_vary_partition_size_fix_dataset
import FSL_algorithm.models.simulation.no_privacy.model2_no_privacy_vary_partition_size_fix_dataset as fsl_no_privacy_vary_partition_size_fix_dataset

import FSL_algorithm.models.simulation.progressive_approach.model1_alt_vary_partition_size_fix_dataset as psl_alt_vary_partition_size_fix_dataset
import FSL_algorithm.models.simulation.progressive_approach.model2_alt_vary_partition_size_fix_dataset as fsl_alt_vary_partition_size_fix_dataset

import FSL_algorithm.models.simulation.differential_privacy.model1_dp_vary_partition_size_fix_dataset as psl_dp_vary_partition_size_fix_dataset
import FSL_algorithm.models.simulation.differential_privacy.model2_dp_vary_partition_size_fix_dataset as fsl_dp_vary_partition_size_fix_dataset

# distributed implementation
import FSL_algorithm.models.distributed.client_based_DC.model1_alt_vary_partition_size_fix_dataset as psl_alt_vary_partition_size_fix_dataset_dist
import FSL_algorithm.models.distributed.client_based_DC.model2_alt_vary_partition_size_fix_dataset as fsl_alt_vary_partition_size_fix_dataset_dist
# import FSL_algorithm.models.distributed.client_based_DP.model1_dp_vary_partition_size_fix_dataset as psl_dp_vary_partition_size_fix_dataset_dist
# import FSL_algorithm.models.distributed.client_based_DP.model2_dp_vary_partition_size_fix_dataset as fsl_dp_vary_partition_size_fix_dataset_dist
import FSL_algorithm.models.distributed.no_privacy.model1_vary_partition_size_fix_dataset as psl_no_privacy_vary_partition_size_fix_dataset_dist
import FSL_algorithm.models.distributed.no_privacy.model2_vary_partition_size_fix_dataset as fsl_no_privacy_vary_partition_size_fix_dataset_dist
import FSL_algorithm.models.distributed.no_privacy.model1_fix_partition_size_vary_dataset as psl_no_privacy_fix_partition_size_vary_dataset_dist
import FSL_algorithm.models.distributed.no_privacy.model2_fix_partition_size_vary_dataset as fsl_no_privacy_fix_partition_size_vary_dataset_dist

this_dir = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.join(this_dir, '..')
EXP_DIR = os.path.join(this_dir, '..', '..', 'exp')
os.makedirs(EXP_DIR, exist_ok=True)


class Config:
    # attack experiment working directory
    WD = os.path.join(EXP_DIR, "m1_nop_reconstruction_client_20_vary_partition_size_fix_dataset_base_500") 

    # learner experiment parent directory
    PD = EXP_DIR

    # attack experiment intermediate data dir
    INTERMEDIATE_DATA_DIR = "Train/"

    # privacy-aware approach parameter (DC frequency, or EPS)
    PARAM = 2

    # model list
    MODELS = [fsl_no_privacy_fix_partition_size_vary_dataset]

    # MODELS to run:
    # psl_no_privacy_fix_partition_size_vary_dataset
    # fsl_no_privacy_fix_partition_size_vary_dataset
    # psl_no_privacy_vary_partition_size_fix_dataset
    # fsl_no_privacy_vary_partition_size_fix_dataset
    # psl_alt_vary_partition_size_fix_dataset
    # fsl_alt_vary_partition_size_fix_dataset
    # psl_dp_vary_partition_size_fix_dataset
    # fsl_dp_vary_partition_size_fix_dataset

    # distributed models
    # psl_no_privacy_fix_partition_size_vary_dataset_dist
    # fsl_no_privacy_fix_partition_size_vary_dataset_dist
    # psl_no_privacy_vary_partition_size_fix_dataset_dist
    # fsl_no_privacy_vary_partition_size_fix_dataset_dist
    # psl_alt_vary_partition_size_fix_dataset_dist
    # fsl_alt_vary_partition_size_fix_dataset_dist
    # # psl_dp_vary_partition_size_fix_dataset_dist
    # # fsl_dp_vary_partition_size_fix_dataset_dist

    #Number of epochs
    EPOCHS = 20

    #Attack Epoch
    attack_epoch = "9"

    #Number of epochs
    ATTACKER_EPOCHS = 10
    K = 3 #3 --> k=3 50 epoche 0.01
    LAMBDA = 0.9 #0.01

    #Number of clients
    CLIENTS=20
    
    #Number of max_clients (used in Equal Work Clients scenerio)
    MAXCLIENTS=500

    #Bastch size
    BATCH_SIZE= 32

    #Optimizer
    OPTIMIZER ="Adam"

    #Learning rate
    LR = 1e-4

    #Count how many times f1_score greater than f1_scorebest
    MAX_COUNTER = 5

    #Seed #41, 21
    SEED = 41

    #NB: more than one cut works only for model2 at this moment --> like Thor model
    #Thor: the NN is divided in "thor" parts
    THOR = 2 

    #Cuts: index of cuts layer
    CUTS = [0,3]
    
    @staticmethod
    def load_config(dict_conf):
        # load settings from config file
        for k, v in dict_conf.items():
            setattr(Config, k, v)
