import os
from utils import config
import shutil
import argparse
import json

from mipsfusion import MIPSFusion


if __name__ == '__main__':
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])  # dir to save result of this running
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shutil.copy("mipsfusion.py", os.path.join(save_path, 'mipsfusion.py'))
    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = MIPSFusion(cfg)
    slam.run()
