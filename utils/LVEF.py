'''
    Automatic calculation of LVEF for the EchoNet-Dynamic dataset using the Simpson's Rule:
    https://doi.org/10.1161/01.CIR.60.4.760.

    For the CAMUS dataset, it is recommended to use the latest code provided:
    https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8/folder/64b5a9b473e9f00492ce9036.
'''


import cv2 
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def NearestVertex(trg, points):
    point_set = []
    dis0 = np.reshape(trg, (1,-1,2)).repeat(points.shape[0], axis=0)
    dis1 = np.reshape(points, (-1,1,2)).repeat(3, axis=1)
    dis = np.linalg.norm(dis0 - dis1, axis=-1, keepdims=False)
    for i in range(dis.shape[1]):
        point_set.append(points[dis[:,i].argmin()])
    return point_set


def PerpendicularLine(L_p, div_p, mid_p, points):
    L_vec = L_p[0] - L_p[1]
    L_vec = np.reshape(L_vec, (1,1,2)).repeat(points.shape[0], axis=0)

    div_ps = np.reshape(div_p, (1,1,2)).repeat(points.shape[0], axis=0)
    vert_vec = div_ps - points
    
    prod = vert_vec * L_vec
    prod = prod[:,:,0] + prod[:,:,1]
    prod = np.abs(prod)

    inc = np.argsort(prod, axis=0)
    vert_p1 = points[inc[0,0]][0]
    point_set = [vert_p1]

    for i in inc[1:]:
        if (vert_p1[0] - mid_p[0] + 1e-5) * (points[i[0],0,0] - mid_p[0] + 1e-5) < 0:
            point_set.append(points[i[0],0])
            break
    
    return point_set


def get_line(in_trg, out_trg, points):
    base_vec = out_trg[1] - out_trg[2]
    base_vec = np.reshape(base_vec, (1,1,2)).repeat(points.shape[0], axis=0)

    head = np.reshape(in_trg[0], (1,1,2)).repeat(points.shape[0], axis=0)
    vert_vec = head - points

    prod = vert_vec * base_vec
    prod = prod[:,:,0] + prod[:,:,1]
    prod = np.abs(prod)

    inc = np.argsort(prod, axis=0)
    for i in inc[1:]:
        if points[i[0],0,1] >= max(in_trg[1][0,1], in_trg[2][0,1]) and \
           points[i[0],0,0] <= max(in_trg[1][0,0], in_trg[2][0,0]) and \
           points[i[0],0,0] >= min(in_trg[1][0,0], in_trg[2][0,0]):
            return points[i[0]][0]
        

def get_info(img_arr):
    contours, _ = cv2.findContours(img_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    if len(contours) > 1:
        idx = 0
        nmax = 0
        for i in range(len(contours)):
            if contours[i].shape[0] > nmax:
                idx = i
                nmax = contours[i].shape[0]
        contours = contours[idx:idx+1] 

    # smallest peripheral triangle
    _, out_trg = cv2.minEnclosingTriangle(contours[0])
    # inscribed triangle
    in_trg = NearestVertex(out_trg, contours[0])

    out_trg = sorted(out_trg, key=lambda x: x[...,1])
    in_trg = sorted(in_trg, key=lambda x: x[...,1])

    # the longest line of the long-axis section
    inter_p = get_line(in_trg, out_trg, contours[0])
    L_p = [in_trg[0][0], inter_p]

    # trisection point
    div_p1 = (2 * L_p[0] + L_p[1]) / 3.0
    div_p2 = (L_p[0] + 2 * L_p[1]) / 3.0

    # endpoints of perpendicular line through trisection point
    inc = np.argsort(contours[0][...,0], axis=0).squeeze()
    mid_p = (contours[0][inc[-1],0] + contours[0][inc[0],0]) / 2
    div_l1 = PerpendicularLine(L_p, div_p1, mid_p, contours[0])
    div_l2 = PerpendicularLine(L_p, div_p2, mid_p, contours[0])

    # visualization
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR) * 255

    for c in combinations(out_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_arr, start_p, end_p,color=(255,0,0))

    for c in combinations(in_trg, 2):
        start_p = [int(x) for x in c[0][0]]
        end_p = [int(x) for x in c[1][0]]
        cv2.line(img_arr, start_p, end_p,color=(0,255,0))

    start_p = [int(x) for x in L_p[0]]
    end_p = [int(x) for x in L_p[1]]
    cv2.line(img_arr, start_p, end_p,color=(0,0,255))

    start_p = [int(x) for x in div_l1[0]]
    end_p = [int(x) for x in div_l1[1]]
    cv2.line(img_arr, start_p, end_p,color=(0,0,255))

    start_p = [int(x) for x in div_l2[0]]
    end_p = [int(x) for x in div_l2[1]]
    cv2.line(img_arr, start_p, end_p,color=(0,0,255))

    plt.imshow(img_arr)
    plt.show()

    return out_trg, in_trg, contours, L_p, [div_l1, div_l2]


def get_volume(img_arr):
    info = get_info(img_arr)
    div_line1, div_line2 = info[-1]
    L = info[-2][0] - info[-2][1]
    dis1 = div_line1[0] - div_line1[1]
    dis2 = div_line2[0] - div_line2[1]

    # trisecting surface radius
    dis1 = np.linalg.norm(dis1, axis=-1, keepdims=False) / 2.0
    dis2 = np.linalg.norm(dis2, axis=-1, keepdims=False) / 2.0
    if dis1 > dis2:
        dis1, dis2 = dis2, dis1

    # LV length 
    L = np.linalg.norm(L, axis=-1, keepdims=False)

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