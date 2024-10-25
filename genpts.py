import sys
import numpy as np

def get_points(nr_points, minx=0, maxx=1, miny=0, maxy=1):

    # random values between 0 - 1
    points = np.random.rand(2, nr_points)

    # map x and y values between minx - maxx, miny - maxy
    points[0, :] = np.interp(points[0, :], [0, 1], [minx, maxx])
    points[1, :] = np.interp(points[1, :], [0, 1], [miny, maxy])

    return points


if __name__ == '__main__':

    p =  get_points (int(sys.argv[1]))

    f = open(sys.argv[2],"w")

    f.write(sys.argv[1]+"\n")
    
    for i in range(int(sys.argv[1])) :
        f.write(str(p[0,i])+" "+str(p[1,i])+"\n")
        
    f.close()
