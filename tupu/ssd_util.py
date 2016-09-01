import math
import numpy as np
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

def project_angle(r, width, height):
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

def ocr_group(dets):
    def compute_center(x0,y0,x1,y1):
        x_c = (x0 + x1)/2
        y_c = (y0 + y1)/2
        x_d = (x1 - x0)
        y_d = (y1 - y0)
        r = math.sqrt(x_d*x_d + y_d*y_d)/2
        return x_c, y_c , r
    def compute_distance(k,x0,y0, x1, y1):
        b = y0 - (k*x0)
        d = abs(k*x1 - y1 + b)/math.sqrt(k*k + 1)
        return d
    def is_inline(k, a_point, b_point):
        x0, y0, r0 = a_point
        x1, y1, r1 = b_point
        if k is None:
            distance = abs(x1 - x0)
            return distance < r0/2 and distance < r1/2
        distance = compute_distance(k, x0, y0, x1, y1)
        return distance < r0/4 and distance < r1/4
    dets = np.array(dets, np.float32)
    l = dets.shape[0]
    points = [compute_center(dets[i][0], dets[i][1], dets[i][2], dets[i][3]) for i in range(l)]
    point_to_line = [[] for i in range(l)]
    line_to_point = [[set() for k in range(4) ] for i in range(l)]
    for i in range(l):
        cur_degree = dets[i][5]
        k_array = [math.tan(math.radians(degree)) for degree in [cur_degree, cur_degree + 3, cur_degree-3]] \
                  + [None]
        for k_index, k in enumerate(k_array):
            for j in range(l):
                if is_inline(k, points[i], points[j]):
                    line_to_point[i][k_index].add(j)
                    point_to_line[j].append((i, k_index))
    point_left = set([i for i in range(l)])

    res = []
    def x_cmp(a, b):
        x1 = dets[a][0]
        x2 = dets[b][0]
        if x1 < x2:
            return -1
        elif x1 > x2:
            return 1
        return 0
    def y_cmp(a, b):
        y1 = dets[a][1]
        y2 = dets[b][1]
        if y1 < y2:
            return -1
        elif y1 > y2:
            return 1
        return 0
                                       
    while len(point_left) > 0:
        max_len = 0
        max_flag = None
        for i in range(l):
            for k_index, tm_line in enumerate(line_to_point[i]):
                if len(tm_line) > max_len:
                    max_len = len(tm_line)
                    max_flag = (tm_line, i, k_index)
        max_line, max_line_index, max_k_index = max_flag
        max_points = [p for p in max_line]
        for p_index in max_points:
            other_lines = point_to_line[p_index]
            point_left.remove(p_index)
            for i_index, k_index in other_lines:
                line_to_point[i_index][k_index].remove(p_index)
        if max_k_index == 3:
            max_points.sort(y_cmp)
        else:
            max_points.sort(x_cmp)    
        res.append(max_points)
    def line_cmp(a,b):
        assert len(a) > 0 and len(b) > 0
        x_a = dets[a[0]][0]
        y_a = dets[a[0]][1]
        x_b = dets[b[0]][0]
        y_b = dets[b[0]][1]
        if y_a < y_b:
            return -1
        elif y_a > y_b:
            return 1
        elif x_a < x_b:
            return -1
        elif x_a > x_b:
            return 1
        return 0
    res.sort(line_cmp)
        
    return res 
