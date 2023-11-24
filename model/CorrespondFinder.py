import torch
import numpy as np


class CorrespondFinder():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = SLAM.device
        self.dataset = self.slam.dataset
        self.kfSet = self.slam.kfSet

        # TODO


