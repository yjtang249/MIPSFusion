import os
import numpy as np
import torch

from helper_functions.printTime import printCurrentDatetime

class ActiveMap():
    def __init__(self):
        pass


    def run(self):
        print(printCurrentDatetime() + "(Active Mapping process) Process starts!!! (PID=%d)" % os.getpid())

