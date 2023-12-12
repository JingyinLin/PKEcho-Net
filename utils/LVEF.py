import cv2 
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def NearestVertex(trg, points):
    point_set = []
    dis0 = np.reshape(trg,(1,-1,2)).repeat(points.shape[0], axis=0)
    dis1 = np.reshape(points,(-1,1,2)).repeat(3, axis=1)
    dis = np.linalg.norm(dis0 - dis1, axis=-1, keepdims=False)
    for i in range(dis.shape[1]):
        point_set.append(points[dis[:,i].argmin()])
    return point_set


def get_line(line1, line2, p1, points):
    if line1[1] > line2[1]:
        line1, line2 = line2, line1

    line_vec = line1 - line2
    line_vec = np.reshape(line_vec,(1,1,2)).repeat(points.shape[0],axis=0)

    p1_s = np.reshape(p1,(1,1,2)).repeat(points.shape[0],axis=0)
    vert_vec = p1_s - points
    
    prod = vert_vec * line_vec
    prod = prod[:,:,0] + prod[:,:,1]
    prod = np.abs(prod)

    inc = np.argsort(prod, axis=0)
    point1 = points[inc[0,0]][0]
    point_set = [point1]
    for i in inc[1:]:
        if (point1[0] - p1[0]) * (points[i[0],0,0] - p1[0]) < 0:
            point_set.append(points[i[0],0])
            break
    
    return point_set


def get_info(img_arr):
    img_result = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    contours_, _ = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    if len(contours) > 1:
        idx = 0
        nmax = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > nmax:
                idx = i
                nmax = contours[i].shape[0]
        contours = contours[idx:idx+1] 

    if len(contours_) > 1:
        idx = 0
        nmax = 0
        for i in range(len(contours_)):
            if contours_[i].shape[0] > nmax:
                idx = i
                nmax = contours_[i].shape[0]
        contours_ = contours_[idx:idx+1]  

    # inscribed and circumscribed triangle
    _, out_trg = cv2.minEnclosingTriangle(contours_[0])
    in_trg = NearestVertex(out_trg, contours[0])

    for c in combinations(out_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_result, start_p, end_p,color=(255,0,0))

    for c in combinations(in_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_result, start_p, end_p,color=(0,255,0))

    # midpoint of circumscribed triangle
    out_mid = [np.mean(x, axis=0) for x in combinations(out_trg, 2)]
    
    # Euclidean distance from each midpoint to each inscribed triangle vertex
    dis0 = np.reshape(in_trg,(1,3,2)).repeat(3,axis=0)
    dis1 = np.reshape(out_mid,(3,1,2)).repeat(3,axis=1)
    dis = np.linalg.norm(dis0-dis1,axis=-1,keepdims=False)

    # find the longest distance 
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(dis)
    L_p = [in_trg[max_indx[0]][0],out_mid[max_indx[1]][0]]

    start_p = [int(x) for x in L_p[0]]
    end_p = [int(x) for x in L_p[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    # trisection point
    div_p1 = (2*L_p[0] + L_p[1])/3.0
    div_p2 = (L_p[0] + 2*L_p[1])/3.0

    # endpoints of vertical line through trisection point
    div_l1 = get_line(L_p[0], L_p[1], div_p1, contours[0])
    div_l2 = get_line(L_p[0], L_p[1], div_p2, contours[0])

    start_p = [int(x) for x in div_l1[0]]
    end_p = [int(x) for x in div_l1[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    start_p = [int(x) for x in div_l2[0]]
    end_p = [int(x) for x in div_l2[1]]
    cv2.line(img_result, start_p, end_p,color=(0,0,255))

    # plt.imshow(img_result)
    # plt.show()

    return out_trg, in_trg, contours, L_p, [div_l1, div_l2]


def get_volume(img_arr):
    info = get_info(img_arr)
    div_line1, div_line2 = info[-1]
    L = info[-2][0] - info[-2][1]
    dis1 = div_line1[0] - div_line1[1]
    dis2 = div_line2[0] - div_line2[1]

    # trisecting surface radius
    dis1 = np.linalg.norm(dis1,axis=-1,keepdims=False) / 2.0
    dis2 = np.linalg.norm(dis2,axis=-1,keepdims=False) / 2.0
    if dis1 > dis2:
        dis1, dis2 = dis2, dis1

    # LV length 
    L = np.linalg.norm(L,axis=-1,keepdims=False)

    ap = np.pi * dis1 * dis1
    am = np.pi * dis2 * dis2

    v1 = am * L / 3
    v2 = (am + ap) / 2 * L / 3
    v3 = ap * L / 3 / 3
    v = v1 + v2 + v3

    return v


def get_EF(pred_ed, pred_es):
    edv = get_volume(pred_ed)
    esv = get_volume(pred_es)
    
    ef = (edv - esv) / edv * 100

    return ef