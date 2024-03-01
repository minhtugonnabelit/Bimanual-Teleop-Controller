import numpy as np 
from math import pi

import spatialmath as sm
import spatialgeometry as geometry
import roboticstoolbox as rtb
from swift import Swift

import threading
import logging
import time

pr2 = rtb.models.PR2()
env = Swift()



qtest = np.zeros(31)
qtest[16:23] = [-pi/6,pi/6,0,-pi/2,0,-pi/4,pi/2]
qtest[23:30] = [pi/6,pi/6,0,-pi/2,0,-pi/4,pi/2]

env.launch(realtime=True)
env.add(pr2)

traj = rtb.jtraj(pr2.qr, qtest, 100)
for q in traj.q:
    pr2.q = q
    env.step(0.02)

env.hold()