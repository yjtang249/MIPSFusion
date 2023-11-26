import os
from utils import config
import shutil
import argparse
import json

from mipsfusion import MIPSFusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--config', type=str, help='config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])  # dir to save result of this running
    os.makedirs(save_path, exist_ok=True)

    slam = MIPSFusion(cfg)
    slam.run()
