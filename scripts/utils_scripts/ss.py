# Require libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from spatialmath import SE2
from spatialmath.base import *
from roboticstoolbox import DHLink, DHRobot
from ir_support import tranimate_custom 
import roboticstoolbox as rtb
# Useful variables
from math import pi
print(rtb.__version__)
def lab2_starter_run():
    ## Go through 3 parts 
    # Lab2Starter.run_part1()
    Lab2Starter.run_part2()
    Lab2Starter.run_part3()

    ## Class example
    # in your new code file do:
    #   from lab2_starter import *
    # Create the base class from the workspace
    #
    #   obj = Lab2Starter()
    #
    # Run this simple method which shows the values of the properties
    # 
    #   obj.show_values()
    #
    # Get the values of the properties
    # 
    #   print(obj.class_property1)
    #   print(obj.class_property2)
    #   print(obj.class_property3)
    #
    # Change the values of the properties
    #
    #   obj.class_property1 = obj.class_property1 + 1
    #   obj.class_property2 = 'b'
    #   obj.class_property3 = plot(rand(100,1),'r.')
    #   plt.plot(np.random.rand(100), 'r.')
    #   plt.show()
    #
    # Get the values of the properties and see how they have changed
    #
    #   print(obj.class_property1)
    #   print(obj.class_property2)
    #   print(obj.class_property3)
    #
    # Run this simple method which shows the values of the properties
    #
    #   obj.change_values()


# ---------------------------------------------------------------------------------------#
class Lab2Starter:
    ## In Python, it is conventional to define the __init__ (constructor) 
    # method as the first method within a class. This method will run when
    # an instance of the class is created.
    def __init__(self):
        self.class_property1 = 1
        self.class_property2 = 'a'
        self.class_property3 = None
        
    def show_values(self):
        print(self.class_property1)
        print(self.class_property2)
        print(self.class_property3)
        
    def change_values(self):
        self.class_property1 += 1
        self.class_property2 = chr(ord(self.class_property2) + 1)
        if self.class_property3 is not None:
            del self.class_property3
        self.class_property3 = None

    ## Doing transformation (translation and rotation)
    @staticmethod
    def run_part1():
        # Clear the figure in focus or create one
        plt.clf()
        plt.subplot() # Add new axes to the current figure

        ## Transform 1
        # To get transform data of a SE2 object, we use the .A operator
        T1 = SE2(1, 2, 30 * pi / 180).A
        print("T1 = \n",SE2(T1))

        trplot2(T1, frame='1', color='b')
        plt.pause(0.01) # A pause is added to display the plot
        input("Press Enter to continue\n")

        ## Transform 2
        T2 = SE2(2, 1, 0).A
        print("T2 = \n",SE2(T2))

        trplot2(T2, frame='2', color='r')
        plt.pause(0.01)
        input("Press Enter to continue\n")

        ## Transform 3 is T1 * T2
        T3 = T1 @ T2
        print("T3 = \n",SE2(T3))

        trplot2(T3, frame='3', color='g')
        plt.pause(0.01)
        input("Press Enter to continue\n")

        ## Transform 4 is T2 * T1
        T4 = T2 @ T1
        print("T4 = \n",SE2(T4))

        trplot2(T4, frame='4', color='k')
        plt.pause(0.01)
        input("Press Enter to continue\n")

        ## Set up the Axis so we can see everything
        plotvol2(dim = [0,5], equal= True, grid = True)
        input("Press Enter to continue\n")

        ## What is the transform T5 between T3 and T4 (i.e. how can you transform T3 to be T4)?
        T5 = linalg.inv(T3) @ T4
        print("T5 = \n",SE2(T5)) 
        # Since T3 @ linalg.inv(T3) @ T4 - T4 = 0, we know T3 @ T5 = T4
        input("Press Enter to continue\n")

        ## Transform from transform to point
        P = np.array([3,2])
        print("P = ",P)
        plot_point(P, "*", text = "P") # This function plot 2D point only
        plt.pause(0.01)

        P1 = linalg.inv(T1) @ np.append(P,1) 
        print("P1 =\n", h2e(linalg.inv(T1) @ e2h(P)))
        # More compact
        print(homtrans(linalg.inv(T1), P))
        
        # In respect to T2
        P2 = homtrans(linalg.inv(T2), P)
        print("P2 = \n", P2)
        input("Press Enter to continue\n")

        ## Page 27
        plt.close('all') # Close the current figure (because 3D axes can't be added into 2D figure)
        plotvol3(dim = [-2,2], equal= True, grid = True) # Generate new 3D figure
        
        R = rotx(pi/2)
        print("R = \n", np.round(R,decimals=4))
        print("## Page 27")

        # Show the trplot as 'rviz'
        trplot(R, frame = "1", color = "rgb", style= "rviz")
        plt.pause(0.01)
        input("Press Enter to continue\n")

        # Show the trplot as 'arrow'
        plt.cla() # Clear the current axes
        plotvol3(dim = [-2,2], equal= True, grid = True)
        trplot(R, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
        plt.pause(0.01)
        input("Press Enter to continue\n")
        # Incrementing by 1 deg
        for i in range(1,91):
            plt.cla()
            plotvol3(dim = [-2,2], equal= True, grid = True)
            R = R @ rotx(pi/180)
            trplot(R, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            plt.draw()
            plt.pause(0.01)
        input("Press Enter to continue\n")

        ## Creating a new one from scratch
        for i in range(0,92,2):
            plt.cla() # Reset the figure axes
            plotvol3(dim = [-2,2], equal= True, grid = True)
            R = rotx(i * pi/180)
            print("R =\n", R)
            trplot(R, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            plt.draw()
            plt.pause(0.01)
        input("Press Enter to continue\n")

        ## Rotate back and forwards around each of the 3 axes
        # Incrementing by 1 deg 
        R = np.eye(3)
        rotation_axis = 2
        loop_elements = np.concatenate((np.arange(0, 92, 2), np.arange(88, -2, -2))) # an array from 0 to 90 by a step of 2, then reverse to 0 again 
        for i in loop_elements:
            # Reset the figure axes        
            plt.cla()
            plotvol3(dim = [-2,2], equal= True, grid = True)

            if rotation_axis == 1: # X axis rotation
                R = rotx(i * pi/180)
            elif rotation_axis == 2: # Y axis rotation
                R = roty(i * pi/180)
            else: # rotation_axis ==3 # Z axis rotation
                R = rotz(i * pi/180)

            trplot(R, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            plt.draw()
            plt.pause(0.01)
        input("Press Enter to continue\n")
        
        ## Rotate all 3 axis at the same time
        plt.cla()
        plotvol3(dim = [-2,2], equal= True, grid = True)
        R = np.eye(3)
        points = [] # This list is to store all points
        for i in loop_elements:
            plt.cla() # Reset the figure axes
            plotvol3(dim = [-2,2], equal= True, grid = True)
            ax = plt.subplot() # Get the current axes

            R = rotx(i * pi/180) @ roty(i * pi/180) @ rotz(i * pi/180)
            trplot(R, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)            
            sum_of_axes = R[:,0] + R[:,1] + R[:,2]
            point = [sum_of_axes[0], sum_of_axes[1], sum_of_axes[2]] 

            # This below loop is to keep all the points displayed so we can see the path the point moves 
            points.append(point)
            for point in points:
                ax.plot(point[0],point[1],point[2], '*')    
            plt.draw()
            plt.pause(0.01)
        input("Press Enter to continue\n")
        
        ## Questions: Will the plot of the sum of the axis vectors in the orientation be the same no matter the order
        # The coloured point represent the sum of the 3 axis for different
        # rotations. Which plot is which?
        # a. red,green,blue,black = Rotations orders (x,y,z),(z,y,x),(x,z,y),(z,x,y)
        # b. red,green,blue,black = Rotations orders (z,y,x),(x,z,y),(z,x,y),(x,y,z)
        # c. red,green,blue,black = Rotations orders (x,z,y),(z,x,y),(x,y,z),(z,y,x)
        # d. red,green,blue,black = Rotations orders (z,x,y),(x,y,z),(z,y,x),(x,z,y)
        points1 = [] # List to store all points 1
        points2 = [] # List to store all points 2
        points3 = [] # List to store all points 3
        points4 = [] # List to store all points 4
        for i in loop_elements:
            plt.cla() # Reset the figure axes
            plotvol3(dim = [-2,2], equal= True, grid = True)
            ax = plt.subplot() # Get the current axes
            
            R1 = rotx(i * pi/180) @ roty(i * pi/180) @ rotz(i * pi/180)
            R2 = rotz(i * pi/180) @ roty(i * pi/180) @ rotx(i * pi/180)
            R3 = rotx(i * pi/180) @ rotz(i * pi/180) @ roty(i * pi/180)
            R4 = rotz(i * pi/180) @ rotx(i * pi/180) @ roty(i * pi/180)
            trplot(R1, frame = "1", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            trplot(R2, frame = "2", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            trplot(R3, frame = "3", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            trplot(R4, frame = "4", color = "rgb", style= "arrow", width = 0.8, originsize = 0)
            sum_of_axes1 = R1[:,0] + R1[:,1] + R1[:,2]
            sum_of_axes2 = R2[:,0] + R2[:,1] + R2[:,2]
            sum_of_axes3 = R3[:,0] + R3[:,1] + R3[:,2]
            sum_of_axes4 = R4[:,0] + R4[:,1] + R4[:,2]
            point1 = [sum_of_axes1[0], sum_of_axes1[1], sum_of_axes1[2]]
            point2 = [sum_of_axes2[0], sum_of_axes2[1], sum_of_axes2[2]]
            point3 = [sum_of_axes3[0], sum_of_axes3[1], sum_of_axes3[2]]
            point4 = [sum_of_axes4[0], sum_of_axes4[1], sum_of_axes4[2]]
            points1.append(point1)
            points2.append(point2)
            points3.append(point3)
            points4.append(point4)

            # This below loop is to keep all the points displayed so we can see the path the point moves 
            for j in range(len(points1)):
                ax.plot(points1[j][0], points1[j][1], points1[j][2],'r*')
                ax.plot(points2[j][0], points2[j][1], points2[j][2],'g*')
                ax.plot(points3[j][0], points1[j][1], points3[j][2],'b*')
                ax.plot(points4[j][0], points1[j][1], points4[j][2],'k*')
            
            plt.draw()
            plt.pause(0.01)

        input("Press Enter to continue\n")    
        # plt.show()
    
    # ---------------------------------------------------------------------------------------#
    ## Using tranimate_custom to plot frames changing
    @staticmethod
    def run_part2():
        if plt.get_fignums():
            plt.close('all')
            
        tr_origin = np.eye(4)

        rotate_around_x_by_90deg = trotx(pi/2)
        tranimate_custom(tr_origin, rotate_around_x_by_90deg)
        tranimate_custom(rotate_around_x_by_90deg, tr_origin)

        rotate_around_y_by_90deg = troty(pi/2)
        tranimate_custom(tr_origin, rotate_around_y_by_90deg)
        tranimate_custom(rotate_around_y_by_90deg, tr_origin)

        tr1 = transl(0,0,5)
        tranimate_custom(tr_origin, tr1)
        tr2 = transl(5,5,5) @ trotx(pi/4)
        tranimate_custom(tr1, tr2) 

        input("Press Enter to continue\n")

        # Now we start to animate on a fixed axis and see the difference
        dim = [0,10]
        tr1 = transl(0,0,5)
        tranimate_custom(tr_origin, tr1, dim= dim)
        tr2 = transl(5,5,5) @ trotx(pi/4)
        tranimate_custom(tr1, tr2, dim= dim, hold= True) # specify 'True' to keep the frame on figure
        # plt.show()
        input("Press Enter to continue\n")

    # ---------------------------------------------------------------------------------------#
    ## Create a 2 Dof robot, plot at extremities and move the joints using teach
    @staticmethod
    def run_part3():
        plt.close('all')
        link1 = DHLink(d= 0, a= 1.5, alpha= 0, offset= 0, qlim= [-pi/2, pi/2])
        link2 = DHLink(d= 0, a= 0.5, alpha= 0, offset= 0, qlim= [-pi/4, pi/4])
        robot = DHRobot([link1, link2], name = 'RunPart3Robot')
        q = np.zeros([1,2]) # Joints at zero postition
        workspace = [-2, 2, -2, 2, 0, 1]

        robot.plot(q= q, limits= workspace)
        input("Press Enter to continue\n")

        print("The qlim in radians is a NDArray [[min_q1, min_q2], [max_q1, max_q2]]")
        qlim = robot.qlim
        print("q=", qlim)
        
        # Note: Changed this to plot the robot at its minimum qlim
        # vs qlim[1,:] which plotted at q1_max, q2_max
        robot.plot(q = qlim[1,:], limits= workspace)
        input("Press Enter to play with teach and then close the figure to finish\n")
        plt.close() # Close current figure because teach method will create a new one 
        robot.teach(robot.q, limits= workspace) 
        plt.clf()        

# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    lab2_starter_run()