import os
import json

import FSL_algorithm.models.local.no_privacy.model1_no_privacy_equal_work_client as psl_no_privacy_equal_work_client
import FSL_algorithm.models.local.no_privacy.model2_no_privacy_equal_work_client as fsl_no_privacy_equal_work_client
import FSL_algorithm.models.local.no_privacy.model1_no_privacy_equal_work_dataset as psl_no_privacy_equal_work_dataset
import FSL_algorithm.models.local.no_privacy.model2_no_privacy_equal_work_dataset as fsl_no_privacy_equal_work_dataset

import FSL_algorithm.models.local.progressive_approach.model1_alt_equal_work_dataset as psl_alt_equal_work_dataset
import FSL_algorithm.models.local.progressive_approach.model2_alt_equal_work_dataset as fsl_alt_equal_work_dataset

import FSL_algorithm.models.local.differential_privacy.model1_dp_equal_work_dataset as psl_dp_equal_work_dataset
import FSL_algorithm.models.local.differential_privacy.model2_dp_equal_work_dataset as fsl_dp_equal_work_dataset

this_dir = os.path.dirname(os.path.realpath(__file__))
PKG_DIR = os.path.join(this_dir, '..')
EXPR_DIR = os.path.join(this_dir, '..', '..', '..', 'EXPR')
MAIN_DIR = os.path.join(this_dir, '..', '..', '..')


class Config:
    # attack experiment working directory
    WD = "./m1_nop_reconstruction_client_20_equal_work_dataset_base_500_all_samples_all_features_attacker" 
    # attack experiment intermediate data dir
    INTERMEDIATE_DATA_DIR = "Train/"


    # privacy-aware approach parameter (DC frequency or EPS)
    PARAM = 0.1

    # model list
    MODELS = [psl_no_privacy_equal_work_dataset]

    #Number of epochs
    EPOCHS = 20
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
    LR = 3e-4

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