import os
import math
import json
import torch
import pickle
import pandas as pd
import scipy.signal as signal

from parameters import *


def save_as_json(file, data):
    """
    Save data to the json file
    @ params: file: the json file that will be saved to
    @ params: data: data that need to be saved
    """
    data = json.dumps(data)
    with open(file, 'w+') as file:
        file.write(data)
    file.close()


def load_json(file):
    """
    load saved data from json file
    @ params: file: the file that will be loaded
    """
    with open(file, 'r+') as file:
        content = file.read()

    return json.loads(content)


def make_gif(env, f_name):
    command = 'ffmpeg -framerate 5 -i "{tempGifFolder}/step%03d.png" {outputFile}'.format(tempGifFolder=env.gif_folder,
                                                                                          outputFile=f_name)

    os.system(command)

    deleteTempImages = "rm {tempGifFolder}/*".format(tempGifFolder=env.gif_folder)
    os.system(deleteTempImages)
    print("wrote gif")


def check_dir(dirs):
    """
    Check if the dir exists, otherwise create it
    @param dirs:
    @return:
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def create_dirs(params):
    """
    Create all experiment directories
    @param params: class
    @return:
    """
    for k, v in params.__dict__.items():
        if 'PATH' in k.split('_') and len(k.split('_')) == 2:
            check_dir(dirs=v)


def create_config_dict():
    """
    Create config dict to save all params
    @return: dictionary
    """
    params_list = [EXPERIMENT_PARAMS, TRAIN_PARAMS, NETWORK_PARAMS, SUMO_PARAMS]

    config_dict = {}
    for params in params_list:
        params_dict = vars(params)
        for k, v in params_dict.items():
            if not k.startswith('__'):
                config_dict[k] = v

    return config_dict


def create_config_json(path, params):
    """
    create a txt file to record the config dictionary
    @param path: string
    @param params: dictionary, key/param name, value/param value
    """
    file = open(path, 'w')
    for k, v in params.items():
        file.write('{}: {}\n'.format(k, v))
    file.close()


def save_as_csv(file, data):
    """
    save the input data to a csv file
    @param file: file path
    @param data: input data
    @return:
    """
    data = pd.DataFrame(data)
    data.to_csv(file)


def save_as_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2

    return math.sqrt(sum_grad)


def convert_to_item(tensor):
    return tensor.cpu().detach().numpy().item()


def convert_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def convert_to_tensor(data, data_type, device):
    if device is None:
        return torch.as_tensor(data=data,
                               dtype=data_type)
    else:
        return torch.as_tensor(data=data,
                               dtype=data_type,
                               device=device)


def calculate_discount_return(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def set_env(server_number, test=False):
    from env.matsc import MATSC_Env
    return MATSC_Env(server_number, test)


# ARLGOS: the latest training state, written after every update so that a stopped run can
# resume (arlgos/run_unicorn.ps1 -Resume). Upstream's LOAD_MODEL loads model/checkpoint.pkl
# but nothing ever wrote it (UNICORN_SETUP.md, section 4.5). The RNG states go with it, so a
# resumed driver continues its random streams instead of restarting them.
def save_latest(network, curr_episode, model_dir):
    import random
    import numpy as np
    path = os.path.join(model_dir, 'checkpoint.pkl')
    state = {'model_state_dict': network.state_dict(),
             'optimizer_state_dict': [network.actor_optimizer.state_dict(),
                                      network.critic_optimizer.state_dict()],
             'epoch': curr_episode,
             'rng': {'random': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state()}}
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)      # a crash mid-write leaves the previous file intact


def load_latest(network, model_dir):
    """Restore what save_latest wrote; returns the episode counter to continue from."""
    import random
    import numpy as np
    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pkl'), map_location='cpu')
    network.load_state_dict(checkpoint['model_state_dict'])
    network.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'][0])
    network.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'][1])
    rng = checkpoint.get('rng')
    if rng:
        random.setstate(rng['random'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
    return checkpoint['epoch']
