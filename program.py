#
# This is the main program
#
# Author: Steven Guan
#

import math
# The Point_Generator class generates and holds the points (static and moving) 
class Point_Generator():

    def __init__(self):
       self.static_points = []
       self.moving_points = []
       self.isPointGenerated = False
    
    def create_grid(self, side_length, center_point):
        """
        create a grid of points with the center_point at the center of the grid
        """
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
                self.static_point.append((x,y))
                self.moving_points.append([[x,y]])
    
        

    



def main():

    pg = Point_Generator()

    



if __name__ == "__main__":
    main()