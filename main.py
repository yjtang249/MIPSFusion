import os
from utils import config
import shutil
import argparse
import json

from mipsfusion import MIPSFusion


if __name__ == '__main__':
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments for running the NICE-SLAM/iMAP*.')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str, help='output folder, this have higher priority, can overwrite the one in config file')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])  # dir to save result of this running
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    shutil.copy("mipsfusion.py", os.path.join(save_path, 'mipsfusion.py'))
    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = MIPSFusion(cfg)
    slam.run()
