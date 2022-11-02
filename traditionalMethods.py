import numpy as np
import cv2
import utils
import dlib
import os
import sys


# To chheck if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def destDelaunayTriangles(rect, lm_points_2, im_2_copy, draw):
    # Subdiv instance for the rect
    subdiv = cv2.Subdiv2D(rect)

    for i in lm_points_2:
        subdiv.insert(i)

    # print(" Done ")
    triangles_d = subdiv.getTriangleList()
    if draw:
        for t in triangles_d :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                cv2.line(im_2_copy, pt1, pt2, (255, 255, 255), 1)
                cv2.line(im_2_copy, pt2, pt3, (255, 255, 255), 1)
                cv2.line(im_2_copy, pt3, pt1, (255, 255, 255), 1)

        for p in lm_points_2 :
            cv2.circle(im_2_copy, p, 2, (0,0,255), -1)
        cv2.imshow("Delaunay Dest Frame", im_2_copy)
        cv2.waitKey(0)

    return triangles_d


def getTrianglesSrc(dst_triangles, points_1, points_2, im_1, draw):

    src_triangles = []
    # print(points2)
    for i in range(len(dst_triangles)):
        ind = []
        pt1 = (dst_triangles[i][0], dst_triangles[i][1])
        pt2 = (dst_triangles[i][2], dst_triangles[i][3])
        pt3 = (dst_triangles[i][4], dst_triangles[i][5])
        ind.append(points_2.index(pt1))
        ind.append(points_2.index(pt2))
        ind.append(points_2.index(pt3))
        move = [points_1[ind[0]][0], points_1[ind[0]][1], \
                points_1[ind[1]][0], points_1[ind[1]][1], \
                points_1[ind[2]][0], points_1[ind[2]][1]]
        src_triangles.append(move)

    if draw:
        for t in src_triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            cv2.line(im_1, pt1, pt2, (255,255,255), 1)
            cv2.line(im_1, pt2, pt3, (255,255,255), 1)
            cv2.line(im_1, pt1, pt3, (255,255,255), 1)
        cv2.imshow("Delaunay Source Frame", im_1)
        cv2.waitKey(1)

    return np.asarray(src_triangles)

# <Warp Delaunay: In Collaboration with Abhishek Nalawade>
def boundingRect(points):
    rect = list()
    x = np.array([points[0], points[2], points[4]])
    y = np.array([points[1], points[3], points[5]])
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    rect = [xmin, ymin, xmax, ymax]
    return rect

def getInternalCoordinates(barycentricCoor, coor):
    sum = np.sum(barycentricCoor, axis = 0)
    sum = np.round(sum, 4)
    sum_greater_than_zero = sum>0
    sum_less_than_one = sum<=1
    sum = sum_greater_than_zero * sum_less_than_one     # logical and operation
    # print(sum)

    alpha = barycentricCoor[0,:]
    # print(alpha)
    alpha_greater_than_zero = alpha>=-0.00001
    alpha_less_than_one = alpha<=1
    alpha = alpha_less_than_one * alpha_greater_than_zero       # logical and operation
    # print(alpha)

    beta = barycentricCoor[1,:]
    beta_greater_than_zero = beta>=-0.00001
    beta_less_than_one = beta<=1
    beta = beta_less_than_one * beta_greater_than_zero       # logical and operation
    # print(beta)

    gamma = barycentricCoor[2,:]
    gamma_greater_than_zero = gamma>=-0.00001
    gamma_less_than_one = gamma<=1
    gamma = gamma_less_than_one * gamma_greater_than_zero       # logical and operation
    # print(gamma)

    internal_coor = alpha * beta * gamma * sum              # logical and operation
    # print(internal_coor)
    internal_points = barycentricCoor[:, internal_coor]
    dst_internal_points = coor[:, internal_coor]
    return internal_points, dst_internal_points

def bilinearInterpolation(cor, img):
    sh = np.shape(cor)
    # print("Cor  ",cor)
    # print("Cor shape ",sh)
    # print("Im shape ",img.shape)

    pix_val = np.zeros((sh[1],3))
    cor_xy = cor[:2,:]

    up_x = np.ceil(cor[0,:]).astype(np.uint64)
    up_y = np.ceil(cor[1,:]).astype(np.uint64)

    up_x[up_x>=img.shape[1]] = img.shape[1] - 1
    up_y[up_y>=img.shape[0]] = img.shape[0] - 1

    down_x = np.floor(cor[0,:]).astype(np.uint64)
    down_y = np.floor(cor[1,:]).astype(np.uint64)

    a = cor_xy[0,:] - down_x
    b = cor_xy[1,:] - down_y

    wt_top_right = (a*b).reshape((sh[1],1))
    wt_top_left = ((1-a)*b).reshape((sh[1],1))
    wt_down_left = ((1-a)*(1-b)).reshape((sh[1],1))
    wt_down_right = (a*(1-b)).reshape((sh[1],1))


    wt_top_right = np.repeat(wt_top_right, 3, axis=1)
    wt_top_left = np.repeat(wt_top_left, 3, axis=1)
    wt_down_left = np.repeat(wt_down_left, 3, axis=1)
    wt_down_right = np.repeat(wt_down_right, 3, axis=1)

    pix_val = (wt_top_right*img[up_y[:],up_x[:]]) + (wt_top_left*img[up_y[:],down_x[:]]) + \
            (wt_down_left*img[down_y[:],down_x[:]]) + (wt_down_right*img[down_y[:],up_x[:]])

    pix_val[pix_val>255] = 255
    pix_val = pix_val.astype(np.uint8)
    # print(" Pixel val: ", pix_val.shape)
    return pix_val


def warpDel(im_1, im_2, s_triangles, d_triangles, h_2):
    before = im_2.copy()
    # print(" Src Tri ", s_triangles)
    # print(" Dest Tri ", d_triangles)
    # print(" Dest Hull ", h_2)
    for i in range(len(d_triangles)):
        corners = d_triangles[i]
        rect = boundingRect(corners)
        B = np.array([[corners[0], corners[2], corners[4]],[corners[1], corners[3], corners[5]],[1, 1, 1]])
        B_inv = np.linalg.inv(B)
        x = np.arange(rect[0]-1, rect[2]+1)
        y = np.arange(rect[1]-1, rect[3]+1)
        x_mesh = np.repeat(x, y.shape[0])
        y_mesh = np.tile(y, x.shape[0])
        x_mesh = np.reshape(x_mesh, (1, x_mesh.shape[0]))
        y_mesh = np.reshape(y_mesh, (1, y_mesh.shape[0]))
        # print(x_mesh.shape, "  ", y_mesh.shape)
        coor = np.concatenate((x_mesh, y_mesh, np.ones((1,x_mesh.shape[1]))), axis=0)
        barycentricCoor = np.dot(B_inv, coor)
        # print(barycentricCoor)
        internal_points, im_2_internal_points = getInternalCoordinates(barycentricCoor, coor)
        im_2_internal_points = im_2_internal_points.astype(np.int64)

        im_1_corners = s_triangles[i]
        A = np.array([[im_1_corners[0], im_1_corners[2], im_1_corners[4]],[im_1_corners[1], im_1_corners[3], im_1_corners[5]],[1,1,1]])
        im_1_internal_points = np.dot(A, internal_points)
        im_1_internal_points = im_1_internal_points/im_1_internal_points[2]

        pixel_values = bilinearInterpolation(im_1_internal_points, im_1)

        # bilinear Interpolation
        im_2[im_2_internal_points[1], im_2_internal_points[0]] = pixel_values
        # cv2.imshow("warp",im_2)
        # cv2.waitKey(50)
        # break

    rec = cv2.boundingRect(h_2)
    center = ((rec[0] + int(round(rec[2]/2)), rec[1] + int(round(rec[3]/2))))
    mask = np.zeros((im_2.shape[0], im_2.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [h_2], 255)
    dst = cv2.seamlessClone(im_2, before, mask, center, cv2.NORMAL_CLONE)
    # cv2.imshow("Delaunay Warped", dst)
    # print("Distay..")
    return dst



def delaunayWarp(im_1, im_2, points_1, points_2, hulls_2, warpingMethod, draw):
    # Dst image and hull points to get the triangles
    rect = (0, 0, im_2.shape[1], im_2.shape[0])
    # h_1 = hull_lists[0][0]
    h_2 = hulls_2
    lm_points_2 = []
    for p in points_2:
        lm_points_2.append((int(p[0]), int(p[1])))
    d_triangles = destDelaunayTriangles(rect, lm_points_2, im_2.copy(), draw=False)
    s_triangles = getTrianglesSrc(d_triangles, points_1, lm_points_2, im_1.copy(), draw=False)
    dest_op = warpDel(im_1, im_2, s_triangles, d_triangles, np.asarray(h_2))
    return dest_op


def getLHSmat(X,Y):
    #dest feature point coors
    # X, Y = points_2[:,0], points_2[:,1]
    # print(" Dest pts: ", points_2)
    # print(" X: ", X)
    X, Y = np.reshape(X, (1,X.shape[0])), np.reshape(Y, (1,Y.shape[0]))
    # print(" X Shape: ", X.shape)
    z = np.zeros((X.shape[1], 1))
    k_x, k_y = X + z, Y + z
    # print(" k_x: ", k_x)
    r = np.square((k_x - X.T)) + np.square((k_y - Y.T))
    # print(" k_x - Xt: ", k_x - X.T)
    # print(" r: ", r)
    #replacing diagonal elements
    r[r == 0] = 1
    K = r * np.log(r)
    # print(" Kernel: ", K)
    one_t = np.ones((X.shape[1], 1))
    P = np.concatenate((X.T, Y.T, one_t), axis=1)
    upper_mat_K_P = np.concatenate((K, P), axis=1)
    lower_mat_Pt_z = np.concatenate((P.T, np.zeros((3,3))), axis=1)
    # P_T = np.concatenate((X, Y, one_t.T), axis=0)
    # zer = np.zeros((3,3))
    # P_T = np.concatenate((P_T, zer), axis=1)
    LHSmat = np.concatenate((upper_mat_K_P, lower_mat_Pt_z), axis=0)
    # print(" Mat LHS:", LHSmat.shape)

    return LHSmat

def xParams(lhsMat, x_s, lmd):
    #v_1 to v_p + ax,ay,a1 zeros = v_vector_rhs
    x_s = np.reshape(x_s, (x_s.shape[0],1))
    x_s = np.concatenate((x_s, np.zeros((3,1))), axis=0)
    # print("X src:", x_s)
    lhsMat = lhsMat + (lmd * np.eye(lhsMat.shape[0]))
    x_spline = np.linalg.inv(lhsMat)@x_s
    return x_spline

def yParams(lhsMat, y_s, lmd):
    #v_1 to v_p + ax,ay,a1 zeros = v_vector_rhs
    y_s = np.reshape(y_s, (y_s.shape[0],1))
    y_s = np.concatenate((y_s, np.zeros((3,1))), axis=0)
    # print("Y src:", y_s)
    lhsMat = lhsMat + (lmd * np.eye(lhsMat.shape[0]))
    y_spline = np.linalg.inv(lhsMat)@y_s
    return y_spline

def warpedK(control, X, Y):
    cX = np.reshape(control[:,0], (1,control.shape[0]))
    cY = np.reshape(control[:,1], (1,control.shape[0]))
    # print(cX)
    # print(cX-X)
    r = np.square((cX - X)) + np.square((cY - Y))
    # r = ((cX - X)**2 + (cY - Y)**2)
    y, x = np.where(r==0)
    r[y,x] = 1
    ln = np.log(r)
    kernel = r * ln
    # print(kernel)
    return kernel

def thinplatesplineWarp(im_1, im_2, points_1, points_2, hulls_2, save_flag, warpingMethod, draw):
    X_src = points_1[:,0]
    Y_src = points_1[:,1]
    X_dest = points_2[:,0]
    Y_dest = points_2[:,1]
    lambd = 0.0000001
    LHSmat = getLHSmat(X_dest, Y_dest)
    Xparam = xParams(LHSmat, X_src, lambd)
    Yparam = yParams(LHSmat, Y_src, lambd)
    #Destination frame mask
    mask = np.zeros((im_2.shape[0], im_2.shape[1]), dtype=np.uint8)
    # print("Hull in tps:", hulls_2)
    cv2.fillPoly(mask, hulls_2, 255)
    # cv2.imshow("mask", abs)
    # cv2.waitKey(0)
    Y, X = np.where(mask==255)
    X = np.reshape(X, (X.shape[0],1))
    Y = np.reshape(Y, (Y.shape[0],1))
    K = warpedK(points_2, X, Y)
    K = np.concatenate((K, X, Y, np.ones((X.shape[0],1))), axis=1)
    x_dash = K@Xparam
    y_dash = K@Yparam
    x_dash[x_dash < 0] = 0
    y_dash[y_dash < 0] = 0
    x_dash[x_dash > im_1.shape[1]] = im_1.shape[1] - 1
    y_dash[y_dash > im_1.shape[0]] = im_1.shape[0] - 1

    face_bb = cv2.boundingRect(np.asarray(hulls_2))
    center = ((face_bb[0] + int(round(face_bb[2]/2)), face_bb[1] + int(round(face_bb[3]/2))))
    face = np.concatenate((x_dash, y_dash), axis=1)

    pixels = bilinearInterpolation(face.T, im_1)
    before = im_2.copy()
    im_2[Y[:,0], X[:,0]] = pixels
    im_2 = cv2.seamlessClone(im_2, before, mask, center, cv2.NORMAL_CLONE)
    # cv2.imshow("dest warped", im_2)
    # cv2.waitKey(0)
    return im_2



def traditionalMethods(im_1, im_2, points_1, points_2, hulls_2, mode, save_flag, method, draw):
    method = method.lower()
    # print(" Traditional method: "+str(method))
    # hulls_1, hulls_2 = hull_lists
    # print("Hull Dest in trad methods: ", hulls_2)
    if method == 'delaunay':
        # print(" Chosen Method: 'Delaunay' ")
        output = delaunayWarp(im_1, im_2, points_1, points_2, hulls_2, method, draw)
        # cv2.imshow("Delaunay warped", output)
        # cv2.waitKey(0)
    elif method == 'tps':
        output = thinplatesplineWarp(im_1, im_2, np.asarray(points_1), np.asarray(points_2), hulls_2, save_flag, method, draw)
        # cv2.imshow("TPS warped", output)
        # cv2.waitKey(0)
    else:
        print(" Invalid Warping Method Input ... Aborting ")
        sys.exit()

    return output
