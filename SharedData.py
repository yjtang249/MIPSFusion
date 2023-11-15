import torch
import copy

from model.scene_rep import JointEncoding


class SharedData():
    def __init__(self, config, SLAM):
        self.config = config
        self.slam = SLAM
        self.device = self.slam.device

        self.shared_model = None
        self.shared_model_flag = torch.zeros((1, )).share_memory_()  # 0: nothing; 1: active_2_inactive; -1: inactive_2_active


    def send_model_a2i(self, model):
        self.shared_model = copy.deepcopy(model).to(self.device).share_memory()
        self.shared_model_flag[0] = 1

    def send_model_i2a(self, model):
        self.shared_model = copy.deepcopy(model).to(self.device).share_memory()
        self.shared_model_flag[0] = -1
