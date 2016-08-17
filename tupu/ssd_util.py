import math
# shunshizhen
def rotate_point(x, y, x_c, y_c, degree):
   x_new = x - x_c
   y_new = y - y_c
   r = math.radians(degree)
   #print 'degree:',r
   x_r = math.cos(r)*x_new - math.sin(r)*y_new
   y_r = math.cos(r)*y_new + math.sin(r)*x_new
   return x_r + x_c, y_r + y_c

def rotate_rec(x1,y1,x2,y2, degree):
    x_c = (x1 + x2)/2
    y_c = (y1 + y2)/2
    x1_new, y1_new = rotate_point(x1,y1,x_c, y_c, degree)
    x2_new, y2_new = rotate_point(x2,y2,x_c, y_c, degree)
    return x1_new, y1_new, x2_new, y2_new

#print rotate_rec(, 1, 1, -1, 45)

#print rotate_point(2, 0, 1, 0, 120)
def compute_warp_rec(x1, y1, x2, y2, degree):
   points = [(x1, y1), (x2, y2), (x1, y2), (x2, y1)]
   x_c = (x1 + x2)/2
   y_c = (y1 + y2)/2
   rotate_points = [rotate_point(x, y,x_c, y_c, degree) for x, y in points]
   x_min, y_min = rotate_points[0]
   x_max, y_max = rotate_points[0]
   for x, y in rotate_points[1:]:
       x_min = min(x_min, x)
       y_min = min(y_min, y)
       x_max = max(x_max, x)
       y_max = max(y_max, y)
   return x_min, y_min, x_max, y_max

def degree2angle(degree, width, height):
    r = math.radians(degree)
    w = math.cos(r)
    h = math.sin(r)
    w_new = w/width
    h_new = h/height
    l = math.sqrt(w_new*w_new + h_new*h_new)
    return math.asin(h_new/l)

def angle2degree(angle, width, height):
    #s = math.asin(angle)
    x = math.tan(angle)
    x_new = x*height
    l = math.sqrt(x_new*x_new + width*width)
    angle_new = math.asin(x_new/l)
    degree = (angle_new * 180.0)/math.pi
    return degree

def compute_rotate_size(degree, width, height):
    angle = math.radians(degree)
    t = math.tan(abs(angle))
    c = (height - t*width)/(1 - t*t)
    b = width - t*c
    a = t*c
    d = t*b
    w_new = math.sqrt(b*b + d*d)
    h_new = math.sqrt(a*a + c*c)
    return w_new, h_new

def compute_sub_rotate_rec(xmin, ymin, xmax, ymax, degree):
    x_c = (xmin + xmax)/2
    y_c = (ymin + ymax)/2
    width = xmax - xmin
    height = ymax - ymin
    w_new, h_new = compute_rotate_size(degree, width, height)
    x_min_new = x_c - w_new/2
    y_min_new = y_c - h_new/2
    x_max_new = x_c + w_new/2
    y_max_new = y_c + h_new/2
    points = [(x_min_new, y_min_new), (x_max_new, y_min_new), (x_max_new, y_max_new), (x_min_new, y_max_new)]
    return [rotate_point(x, y, x_c, y_c, degree) for x, y in points]    


#width = 3
#height = 7
#degree = -55
#angle = degree2angle(degree, width, height)
#degree_new = angle2degree(angle, width, height)
#print 'angle:%f, degree_new:%f' %(angle, degree_new)

#print degree2angle(45, 2, 1)
#print degree2angle(45, 1, 2)
#print compute_warp_rec(0,0, 2, 2, 45)
#print rotate_point(2,2, 0, 0, 45)
#print rotate_point(2,2, 0, 0,-45)

