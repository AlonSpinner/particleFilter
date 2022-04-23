from dataclasses import dataclass
import numpy as np
from gridFastSlam.geometry import pose2
from gridFastSlam.maps import map2

class particle2:
    def __init__(self, map : map2 ,pose : pose2):
        #particle state
        self.pose = pose
        self.map = map
        
