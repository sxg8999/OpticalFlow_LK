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
    
    def create_grid(self, side_length, center_point):
        """
        create a grid of points with the center_point at the center of the grid
        and return it
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
                static_point.append((x,y))
                moving_points.append([[x,y]])

        return static_points, moving_points
        


class Computer():

    def __init__(self):

        #setup LucasKanade parameters
        self.lkParams = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def calc(self, oldGrayFrame, newGrayFrame, points):
        """
            calculates the optical flow using Lucas Kanade and a modified RANSAC operation
        """

        newPoints,_,_ = cv2.calcOpticalFlowPyrLK(oldGrayFrame, newGrayFrame, points, None, **self.lkParams)
        delta_x, delta_y = Ransac.calc(points, newPoints)
        return newPoints, delta_x, delta_y
        

        

class Ransac():
        
    
    def calc(self, oldList, newList):
        """
            find the optimal movement in the x and y direction 
            and return it
        """

        direction, pointList, size = self.pre_operation(oldList, newList)
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
                sampleCount.append(index)
                currCount += 1
        
        #find the optimal movement/flow
        # dist_1 is the magnitude of movement of a point in the current sample,
        #  while dist_2 is the magnitude of movement from a point in the testing list
        for index in sampleList:
            dist_1,_,_ = pointList[index]
            for item in pointList:
                dist_2,_,_ = item
                diff = abs(dist_1-dist_2)       #get the absolute difference
                if diff < threshold:
                    inlierCount = inlierCount + 1
            if maxInliers < inlierCount:
                maxInliers = inlierCount
                bestModel = index
            
            inlierCount = 0
        _, delta_x, delta_y = pointList[bestModel]
        return delta_x, delta_y

    
    
    def pre_operation(oldList, newList):
        """
            look at all the points and determine the direction of movement
            ***This function is not optimized thus reduces the frame rate signifcantly
        """
        counter_dict = {"UPLEFT":0, "UPRIGHT":0, "DOWNLEFT":0, "DOWNRIGHT":0, "DOWN":0, "UP":0, "LEFT":0, "RIGHT":0}
        list_dict = {"UPLEFT":[], "UPRIGHT":[], "DOWNLEFT":[], "DOWNRIGHT":[], "DOWN":[], "UP":[], "LEFT":[], "RIGHT":[]}

        for (p0,p1) in zip(oldList, newList):
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

        # check confidence rating
        confidence_rating = 0.0
        if total != 0:
            confidence_rating = max_count/total
        
        if(confidence_rating >= confidence_target):
            return best_guess, list_dict[best_guess], counter_dict[best_guess]
        
        return "NONE", [], 0   
        


class Frames():

    def __init__(self):
        self.screen_grabber = mss.mss()
        self.window_size = {"top": 140, "left": 560, "width": 800, "height": 800} 
        self.old_frame = np.array(self.screen_grabber(self.window_size))
        self.old_gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
    
    def update_old(self):
        """
        updates the old frame with the new frame
        """
        self.old_gray_frame = self.new_gray_frame.copy()
    
    def  getNext(self):
        """
        grab another frame
        """
        self.new_frame = np.array(self.screen_grabber(self.window_size))
        self.new_gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    
    def draw_info_box(self,delta_x, delta_y):
        """
        Draws a general optical flow on the new frame
        """
        #Below are  numbers that can be changed to 
        #ones preference

        #defining the corners to form the info box
        top_corner = (600,600)
        bottom_corner = (750,800)
        self.new_frame = cv2.rectangle(top_corner,bottom_corner,(0,255,0), 3)






class Application():

    def run(self):
        pg = Point_Generator()
        computer = Computer()
        frames = Frames()
        
        static_points, old_moving_points = pg.create_grid(300, (400,400)) 

        while True:

            frames.getNext()
            new_moving_points, delta_x, delta_y = computer.compute(frames.old_gray_frame, frames.new_gray_frame,old_moving_points)
            frames.update_old()
            frames.draw_info_box(delta_x, delta_y)
            old_moving_points = new_moving_points

            cv2.imshow('frame',frames.new_frame)

            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()



def main():
    app = Application()
    app.run()
    print("Application Closed!!!")

    

    



if __name__ == "__main__":
    main()