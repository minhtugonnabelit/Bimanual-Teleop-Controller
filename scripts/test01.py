import numpy
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base import argcheck
from spatialmath import base

from swift import Swift

# Create a PR2 robot object
pr2 = rtb.models.PR2()

env = Swift()
env.launch()  
env.add(pr2)
  
env.hold()