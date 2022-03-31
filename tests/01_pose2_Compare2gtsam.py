import numpy as np
from particleFilter.geometry import pose2
import gtsam #pip install gtsam

print('-------------Compose:')
x0_gt = gtsam.Pose2(0,0,0)
dx_turn_gt = gtsam.Pose2(0,0,np.pi/2)
dx_straight_gt = gtsam.Pose2(1,0,0)
x1_gt = x0_gt.compose(dx_turn_gt).compose(dx_straight_gt)

x0 = pose2(0,0,0)
dx_turn = pose2(0,0,np.pi/2)
dx_straight = pose2(1,0,0)
x1 = x0+dx_turn+dx_straight

print(f'x1_gt = \n\t{x1_gt}',end='')
print(f'x1 = \n\t{x1}')

#-----------------------------------------

print('----------Inverse:')
print(f'x1_gt.inverse() = \n\t{x1_gt.inverse()}',end='')
print(f'x1.inverse() = \n\t{x1.inverse()}')

#-----------------------------------------

print('----------Between:')
x2_gt = x1_gt.compose(dx_turn_gt).compose(dx_straight_gt)
x2mx1_gt = gtsam.Pose2.between(x2_gt,x1_gt)

x2 = x1+dx_turn+dx_straight
x2mx1 = x2-x1
print(f'x2mx1_gt = \n\t{x2mx1_gt}',end='')
print(f'x2mx1 = \n\t{x2mx1}')


