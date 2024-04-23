import random
import os
import numpy as np
import yaml
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def open_yaml(path):
    assert ('.yaml' in path) or ('.yml' in path), 'Error : yaml(yml) 파일이 아닌 파일을 열려고 합니다.'
    if os.path.exists(path):
        with open(fr'{path}') as f:
            yaml_file = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_file


def save_yaml(path, yaml_file:dict):
    assert ('.yaml' in path) or ('.yml' in path), 'Error : yaml(yml) 파일이 아닌 파일을 저장하려고 합니다.'
    with open(fr'{path}', 'w') as p:
        yaml.dump(yaml_file, p)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(save_path, model, distributed=False):
    assert save_path.endswith('.pt'), '저장할 모델의 확장자가 .pt로 끝나야합니다.'
    if distributed:
        torch.save(model.module.state_dict(), f'{save_path}')
    else:
        torch.save(model.state_dict(), f'{save_path}')