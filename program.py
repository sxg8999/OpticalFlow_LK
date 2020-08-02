#
# This is the main program
#
# Author: Steven Guan
#

import math
import time
import cv2
import mss
import numpy as np
import random



# The Point_Generator class generates the points of interest
# Other methods of generating points of interest can be used too
class PointGenerator():
    """
    A class used to generate a grid point
    """
    
    def create_grid(self, side_length: int, center_point: tuple):
        """
        Creates a grid of points with the center_point at the center of the grid
        and return it

        Parameters
        ----------
        side_length: int
            The side length of the the region of points (length and width of region)
        center_point: tuple
            The origin of the point grid system(the center of the region). The points
            would be created around the center point

        Returns
        -------
        static_points
            a list of points that is used for reference purposes and doesn't move
        
        initial_points
            a numpy array of feature points 

        """

        static_points = []
        moving_points = []

        half_len = math.floor(side_length/2)
        c_x,c_y = center_point
        top_left_x = c_x - half_len
        top_left_y = c_y - half_len

        offset = 20 # the gap between points
        MAX_POINT_PER_ROW = math.floor(side_length/offset) + 1
        print(MAX_POINT_PER_ROW)
        for i in range(MAX_POINT_PER_ROW):
            for j in range(MAX_POINT_PER_ROW):
                x = top_left_x + (j*offset) 
                y = top_left_y + (i*offset)

                # print(str(x) + ", " + str(y))
                static_points.append((x,y))
                moving_points.append([[x,y]])

        inital_points = np.array(moving_points, dtype = np.float32)

        return static_points, initial_points
        


class Computer():
    """
    A wrapper class for calculating the optical flow of the points

    Attributes
    ----------
    ransac : Ransac
        A class that help filter out outliers and calculates a optimal answer
    """

    def __init__(self):
        """
        Initial setup
        """

        #setup LucasKanade parameters
        self.lkParams = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.ransac = Ransac()

    def calc(self, old_gray_frame, new_gray_frame, points):
        """
        Calculates the optical flow using Lucas-Kanade with a modified RANSAC function

        Parameters
        ----------
        old_gray_frame
            The previous frame that is grayscaled
        new_gray_frame
            A new frame that is grayscaled
        points
            The feature points

        Returns
        -------
        new_points
            New points that are calculated using optical flow analysis on the feature points
        delta_x
            Change of movement/flow in the x direction
        delta_y
            Change of movement/flow in the y direction
        """

        new_points,status,error = cv2.calcOpticalFlowPyrLK old_gray_frame, new_gray_frame, points, None, **(self.lkParams))
        delta_x, delta_y = self.ransac.calc(points, new_points)
        return new_points, delta_x, delta_y
        

        

class Ransac():
        
    
    def calc(self, old_points_list: list, new_points_list: list):
        """
        Remove outliers and find the optimal movement in the x and y direction

        Parameters
        ----------
        old_points_list
            The feature point list of the old frame
        new_points_list
            The points calculated with optical flow
        
        Returns
        -------
        delta_x
            Changes in the x direction
        delta_y
            Changes in the y direction

        """

        # size: is the amount of points that are going in the same general direction
        direction, points_list, size = self.pre_operation(old_points_list, new_points_list)

        #if there is not enough points to determine a general directions
        #then consider there to be no change between the two frames
        if direction == "NONE" or size < 50:
            return 0, 0

        threshold = 2 # a number that is not too small or too large
        bestModel = 0
        maxInliers = 0
        inlierCount = 0
        currCount = 0
        sampleCount = 6     #the number of sample you are testings to determine the optimal change in movement/flow
                            #the higher the number the better the accuracy but comes at the cost of lower frame rate
        sampleList = []

        while currCount < sampleCount:
            index = random.randint(0,size - 1)
            if(sampleList.__contains__(index) == False):
                sampleList.append(index)
                currCount += 1
        
        #find the optimal movement/flow
        # dist_1 is the magnitude of movement of a point in the current sample,
        #  while dist_2 is the magnitude of movement from a point in the testing list
        for index in sampleList:
            dist_1,_,_ = points_list[index]
            for item in points_list:
                dist_2,_,_ = item
                diff = abs(dist_1-dist_2)       #get the absolute difference
                if diff < threshold:
                    inlierCount = inlierCount + 1
            if maxInliers < inlierCount:
                maxInliers = inlierCount
                bestModel = index
            
            inlierCount = 0
        _, delta_x, delta_y = points_list[bestModel]
        return delta_x, delta_y

    
    
    def pre_operation(self, old_points_list: list, new_points_list: list):
        """Isolate the good points from the bad points
        Look at all the points and determine the direction of movement
        ***This function is not optimized thus reduces the frame rate signifcantly

        Parameters
        ----------
        old_points_list
            The feature point list of the old frame
        new_points_list
            The points calculated with optical flow

        
        Returns
        -------
        best_guess
            The best guessed direction of movement
        best_points_list
            All the points that agrees with the guess of the direction
        num_of_points
            The number of points in the best_points_list
        
        """
        counter_dict = {"UPLEFT":0, "UPRIGHT":0, "DOWNLEFT":0, "DOWNRIGHT":0, "DOWN":0, "UP":0, "LEFT":0, "RIGHT":0}
        list_dict = {"UPLEFT":[], "UPRIGHT":[], "DOWNLEFT":[], "DOWNRIGHT":[], "DOWN":[], "UP":[], "LEFT":[], "RIGHT":[]}

        for (p0,p1) in zip(old_points_list, new_points_list):
            x0,y0 = p0.ravel()
            x1, y1 = p1.ravel()
            diff_x = x1 - x0
            diff_y = y1 - y0
            #compute distance between the old point and the new point
            #a^2 + b^2
            # sum_of_sqdiff = math.pow(diff_x,2) + math.pow(diff_y,2) 
            sum_of_sqdiff = (diff_x * diff_x) + (diff_y * diff_y)
            distance = math.sqrt(sum_of_sqdiff)

            #create a tuple containing the distance between old point and new point and
            #also the deltas(change) for x and y
            tmpTuple = (distance, diff_x, diff_y)

            direction_str = ""
            if diff_y < 0:
                direction_str = direction_str + "UP"
            elif diff_y > 0:
                direction_str = direction_str + "DOWN"
            
            if diff_x < 0:
                direction_str = direction_str + "LEFT"
            elif diff_x > 0:
                direction_str = direction_str + "RIGHT"
            
            if(direction_str != ""):
                counter_dict[direction_str] += 1
                list_dict[direction_str].append(tmpTuple)

         #Use the results and formulate a best guess for the general direction

        #criteria:
        # - best guess for general direction is the direction with atleast 60 percent confidence rating
        # - if no direction meets this criteria then the general direction is NONE ( no movement )

        confidence_target = 0.60

        #find the total
        total = 0
        max_count = 0
        best_guess = ""
        for key in counter_dict:
            val = counter_dict[key]
            total += val
            if max_count < val:
                max_count = val
                best_guess = key

        
        confidence_rating = 0.0
        

        # check confidence rating
        if total != 0:
            confidence_rating = max_count/total
        
        if(confidence_rating >= confidence_target):
            best_points_list = list_dict[best_guess]
            num_of_points = counter_dict[best_guess]
        else:
            best_guess = "NONE"
            best_points_list = []
            num_of_points = 0

        return best_guess, best_points_list, num_of_points  
        


class Frames():
    """
    A class that grabs, updates, and draw on frames

    Attributes
    ----------
    screen_grabber
        A object that screenshots a region of the screen
    window_size : dict
        A dictionary that defines the desired region of the screen
    old_frame
        The previous frame
    old_gray_frame
        A grayscaled version of the previous frame
    new_frame
        The newly grabbed frame, or the current frame
    new_gray_frame
        A grayscaled version of the newly grabbed frame

    """


    def __init__(self):

        self.screen_grabber = mss.mss()
        self.window_size = {"top": 140, "left": 560, "width": 800, "height": 800} 
        self.old_frame = np.array(self.screen_grabber.grab((self.window_size)))
        self.old_gray_frame = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        
    
    def update_old(self):
        """
        Updates the old frame with the new frame
        """
        self.old_gray_frame = self.new_gray_frame.copy()
    
    def  next(self):
        """
        Grabs another frame and update new_frame and new_gray_frame 
        """
        self.new_frame = np.array(self.screen_grabber.grab((self.window_size)))
        self.new_gray_frame = cv2.cvtColor(self.new_frame, cv2.COLOR_BGR2GRAY)
    
    def draw_info_box(self,delta_x: np.float32, delta_y: np.float32):
        """
        Draws a general optical flow on the new frame

        Parameters
        ----------
        delta_x : np.float32
            The changes in the x direction
        delta_y : np.float32
            The changes in the y direction

        """
        #Below are  numbers that can be changed to 
        #ones preference

        #defining the corners to form the info box
        top_corner = (600,600)
        bottom_corner = (750,800)
        self.new_frame = cv2.rectangle(self.new_frame, top_corner,bottom_corner,(0,255,0), 3)


        #create a static point and a moving point to show the movement
        static_point = (675, 700)  #this was decided to be the origin point (anywhere within the box would do)
        x = math.ceil(static_point[0] + delta_x)
        y = math.ceil(static_point[1] + delta_y)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        self.new_frame = cv2.circle(self.new_frame, static_point, 5, (0,0,255), 2)
        self.new_frame = cv2.circle(self.new_frame, (x,y), 5, (0,255,255), -1)






class Application():


    def run(self):
        pg = PointGenerator()
        computer = Computer()
        frames = Frames()
        
        counter = 0 #This is the frame counter
        static_points, old_moving_points = pg.create_grid(300, (400,400)) 

        while True:

            #Every 10 frames reset the points (This is to get rid of dead/bad tracking points)
            if counter >= 10:
                static_points, old_moving_points = pg.create_grid(300, (400, 400))
                counter = 0

            frames.next()
            counter += 1

            new_moving_points, delta_x, delta_y = computer.calc(frames.old_gray_frame, frames.new_gray_frame,old_moving_points)
            frames.update_old()
            frames.draw_info_box(delta_x, delta_y)
            old_moving_points = new_moving_points

            cv2.imshow('frame',frames.new_frame)

            #27 is the ESC Key (Press it to stop the application)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()



def main():
    app = Application()
    app.run()
    print("Application Closed!!!")

    

    



if __name__ == "__main__":
    main()