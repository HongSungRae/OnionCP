from utils import misc
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(path)

dataset_path = misc.open_yaml(fr'{path}/configuration.yaml')