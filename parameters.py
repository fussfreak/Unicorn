import os
import datetime

date = datetime.datetime.now()


# ARLGOS: per-run settings come from environment variables (arlgos/run_unicorn.ps1),
# because the Ray workers re-import this file in their own processes.
def env_int(name, default):
    return int(os.environ.get(name, default))


# ARLGOS: green / yellow / time-to-teleport per dataset family, from the README
# ("Recommended settings") and paper Sec. V-C.
DATASET_FAMILY = {'grid_network_5_5': 'ma2c', 'monaco_network_30': 'ma2c',
                  'cologne_network_8': 'resco', 'ingolstadt_network_21': 'resco',
                  'arterial_network_4_4': 'resco', 'grid_network_4_4': 'resco',
                  'shaoxing_network_7': 'gesa', 'shenzhen_network_29': 'gesa',
                  'shenzhen_network_55': 'gesa'}
FAMILY_TIMING = {'ma2c': (10, 3, 300), 'resco': (15, 5, -1), 'gesa': (15, 5, 600)}


class INPUT_PARAMS:
    MAX_EPISODES        = env_int('UNICORN_MAX_EPISODES', 1500)  # ARLGOS: paper trains ~1500; upstream 3000
    NUM_META_AGENTS     = env_int('UNICORN_WORKERS', 6)
    OBS_TYPE            = 'V1'  # ['V1', 'V2']
    UPDATE_TYPE         = 'PPO_MAC'
    # ARLGOS: UNICORN_RESUME=1 (arlgos/run_unicorn.ps1 -Resume) continues the run named by
    # UNICORN_EXP_NAME from its model/checkpoint.pkl; upstream False / None
    LOAD_MODEL          = os.environ.get('UNICORN_RESUME') == '1'
    EXPERIMENT_PATH     = ('./Train_MATSC/{}'.format(os.environ.get('UNICORN_EXP_NAME'))
                           if LOAD_MODEL and os.environ.get('UNICORN_EXP_NAME') else None)
    CO_TRAIN            = False


class EXPERIMENT_PARAMS:
    BASE_PATH           = 'Train_MATSC'
    EXPERIMENT_NAME     = (os.environ.get('UNICORN_EXP_NAME') or  # ARLGOS
                           '{}_{}_{}_{}_UNICORN'.format(date.year, date.month, date.day, date.hour))
    EXPERIMENT_PATH     = './{}/{}'.format(BASE_PATH, EXPERIMENT_NAME)
    MODEL_PATH          = EXPERIMENT_PATH + '/model'
    GIFS_PATH           = EXPERIMENT_PATH + '/gifs'
    TRAIN_PATH          = EXPERIMENT_PATH + '/train'
    TRIP_PATH           = EXPERIMENT_PATH + '/trip_info'
    CONFIG_FILE_PATH    = EXPERIMENT_PATH + '/config.json'


class SUMO_PARAMS:
    # MA2C DATASETS: 'grid_network_5_5', 'monaco_network_30',
    # RESCO DATASETS: 'cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4', 'grid_network_4_4'
    # GESA DATASETS: 'shaoxing_network_7', 'shenzhen_network_29', 'shenzhen_network_55'
    ALL_DATASETS        = ['cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4',
                           'grid_network_4_4', 'shaoxing_network_7', 'shenzhen_network_29']
    NET_NAME            = os.environ.get('UNICORN_NET', 'cologne_network_8')  # ARLGOS: upstream 'grid_network_5_5'
    NET_PATH            = './maps/{}'.format(NET_NAME)
    CONFIG_PATH         = NET_PATH + '/{}_config.json'.format(NET_NAME)
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    GUI                 = False
    RANDOM_SEED         = True
    OBS_SHARING         = False
    REWARD_DETECTOR     = True
    REGIONAL_REWARD     = True
    MAX_DISTANCE        = None
    SEED                = 1
    TEST_SEEDS          = [1000 + 100 * i for i in range(10)]  # ARLGOS: evaluator_*.py, start_seed + seed * i
    VAL_SEEDS           = list(range(5042, 5062))  # ARLGOS: reserved for checkpoint selection
    MAX_SUMO_STEP       = 3600
    MAX_TEST_STEP       = 3600
    # MA2C: 5s/2s, RESCO and SG: 10s/3s (5s/2s for arterial4x4), GESA: 10s/5s
    # ARLGOS: the comment above disagrees with the README and the paper; the timing
    # now follows the dataset family (FAMILY_TIMING). Upstream was 15 / 5 / 300.
    GREEN_DURATION, YELLOW_DURATION, TELEPORT_TIME = FAMILY_TIMING[DATASET_FAMILY[NET_NAME]]
    # random route
    END_TIME            = 300
    PERIOD              = 2
    MIN_DIS             = 200
    FRINGE_FACTOR       = 3
    # baseline route for grid 5x5 grid network (default: 1100, 925)
    PEAK_FLOW1          = 1100
    PEAK_FLOW2          = 925
    INIT_DENSITY        = 0
    # baseline route for 30 monaco network (default: 325)
    FLOW                = 325
    # baseline route for 16 singapore network
    SG_FLOW             = 1100


class TRAIN_PARAMS:
    MAX_EPISODE         = INPUT_PARAMS.MAX_EPISODES
    NUM_META_AGENTS     = INPUT_PARAMS.NUM_META_AGENTS
    LOAD_MODEL          = INPUT_PARAMS.LOAD_MODEL
    EXPERIMENT_PATH     = INPUT_PARAMS.EXPERIMENT_PATH
    CONTRASTIVE_LOSS    = False
    NEIGHBOR_REWARD     = False
    WANDB               = False
    USE_GPU             = False  # ARLGOS: CPU torch (UNICORN_SETUP.md); upstream True
    NUM_GPU             = 0  # ARLGOS: upstream 1
    SAVE_MODEL_STEP     = 50  # ARLGOS: for validation-based checkpoint choice; upstream 500
    TRAIN_SEED          = env_int('UNICORN_SEED', 0)  # ARLGOS: seeds the driver (upstream 0) and the workers
    SUMMARY_WINDOW      = 30
    RESET_OPTIM         = False
    RESET_OPTIM_STEP    = 20000


class NETWORK_PARAMS:
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    OBS_TYPE            = INPUT_PARAMS.OBS_TYPE
    UPDATE_TYPE         = INPUT_PARAMS.UPDATE_TYPE
    VL_FACTOR           = 0.5
    EL_FACTOR           = 2e-3
    PL_FACTOR           = 2e-4
    CL_FACTOR           = 1e-5
    CONTRASTIVE_TEMP    = 0.2
    CONTRASTIVE_BATCH   = 256
    GAMMA               = .95
    LAMBDA_ADV          = .98
    A_LR_Q              = 1.e-4
    C_LR_Q              = 2.e-4
    GRAD_CLIP           = 10
    EPS_CLIP            = 0.2
    K_EPOCH             = 6


class JOB_OPTIONS:
    GET_EXPERIENCE       = 1
    GET_GRADIENT         = 2


class COMPUTE_OPTIONS:
    MULTI_THREADED       = 1
    SINGLE_THREADED      = 2


class ALGORITHM_OPTIONS:
    A3C                  = 1
    PPO                  = 2


JOB_TYPE                 = JOB_OPTIONS.GET_EXPERIENCE
COMPUTE_TYPE             = COMPUTE_OPTIONS.SINGLE_THREADED
ALGORITHM_TYPE           = ALGORITHM_OPTIONS.PPO