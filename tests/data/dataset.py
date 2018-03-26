import matplotlib.pyplot as plt
#import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti
basedir = 'E:\dataset'

sequence = '00'
dataset = pykitti.odometry(basedir, sequence)
first_pose = dataset.poses[0]

first_velo = dataset.get_velo(0)

print('\nSequence: ' + str(dataset.sequence))
print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nFirst ground truth pose:\n' + str(first_pose))


f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')

velo_range = range(0, first_velo.shape[0], 500)
ax2.scatter(first_velo[velo_range, 0],first_velo[velo_range, 1],first_velo[velo_range, 2],c=first_velo[velo_range, 3],cmap='gray')
ax2.set_title('First Velodyne scan (subsampled)')
plt.show()
