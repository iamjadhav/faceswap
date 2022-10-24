import numpy as np
import cv2
import utils
import imutils
import dlib
import os
import sys
import argparse
from imutils import face_utils
from traditionalMethods import *
from face_fiducials import *

from api import PRN
from api_next import PRN_
from prNet import prnet
from prNet_ import prnet_


def resize_(image):
    image = imutils.resize(image, width = 320)
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Face Swapping')

    parser.add_argument('--source_frame', default='TestData/Rambo.jpg', type=str,
        help='Path to the Source Frame e.g Data/Avril.jpg OR TestData/Rambo.jpg')

    parser.add_argument('--dst_frame', default='TestData/Test1.mp4', type=str,
        help='Path to the Destination Video Frame e.g Data/Data1.mp4 OR'
        'TestData/Test1.mp4')

    parser.add_argument('--method', default='tps', type=str, help='Warping'
        'Approach to use e.g TPS OR Delaunay Triangulation OR PRNet')

    parser.add_argument('--mode', default='1', type=str, help='Mode of Swapping'
        'e.g 1 for Swapping Source Video Frame with a Destination Image OR'
        '2 for Swapping Frames within a Video')

    parser.add_argument('--draw', default=False, type=bool, help='Triangle'
        'Visualization for frames')

    parser.add_argument('--output', default=False, type=bool, help='To save'
        'the Output or not')

    parser.add_argument('--resize', default=False, type=bool, help='To resize'
        'the Input Frames or not')

    arguments = parser.parse_args()

    src_dir = arguments.source_frame
    dst_dir = arguments.dst_frame
    method = arguments.method
    mode = arguments.mode
    visualization = arguments.draw
    save = arguments.output
    resize = arguments.resize
    output = arguments.resize

    dst_vid_file = os.path.basename(dst_dir)
    # print(dst_vid_file)
    dst_vid = dst_vid_file.split('.')[0]
    # print(dst_vid)

    cap = cv2.VideoCapture(dst_dir)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\n Total Frames in the Video ----->>>>> " + str(length))
    # print("\n")

    _, image = cap.read()

    width = 320
    if resize == True:
        # image = imutils.resize(image, width)
        image = resize_(image)
    height = image[0]
    width = image[1]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if output:
        out = cv2.VideoWriter('{}_Output_{}.avi'.format(method, dst_dir),
                                fourcc, 15, (width, height))

    count = 0

    if mode == "1":
        print(" Mode 1 Selected (Image to Video Frame) ")
        print("\n")
        image_1 = cv2.imread(src_dir)
        if resize == True:
            image = resize_(image_1)

        number_of_faces, im_1_pts, hulls_1 = face_fiducials(image_1)
        print("Total Faces in the Source Image -->> ", number_of_faces)
        # print("Convex Hull List -->> ", hulls_1)
        # sys.exit()

        if number_of_faces != 1:
            print(" More than one face detected ... Aborting !!")
            sys.exit()

        _, image_2 = cap.read()

        width = 320
        if resize == True:
            # image = imutils.resize(image, width)
            image_2 = resize_(image_2)
        height = image_2[0]
        width = image_2[1]

        if method.lower() == 'prnet':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            pNet = PRN(is_dlib = True)

        while(cap.isOpened()):
            count += 1

            ret, image_2 = cap.read()

            if ret == True:

                if resize == True:
                    image = resize_(image_1)

                if method.lower() == 'prnet':
                    pos, output = prnet(pNet, image_2, image_1)
                    if pos is None:
                        continue
                    else:
                        print(" Frame No -->> " + str(count))
                else:
                    print("\n Traditional Approach  --->>> ")
                    # cv2.imshow("Video Frame", image_2)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    number_of_faces, im_2_pts, hulls_2 = face_fiducials(image_2)
                    # print(" Total Faces-(Dest Frame) -->> ", number_of_faces)
                    if number_of_faces == 0:
                        continue
                    else:
                        print(" Total Faces-(Dest Frame) (Not Zero) -->> ", number_of_faces)
                    # hull_lists = [hulls_1, hulls_2]
                    # print("Hull 2 main: ", hulls_2)
                    tradionalOutput = traditionalMethods(image_1, image_2, im_1_pts, im_2_pts, hulls_2, mode, save, method, visualization)

                cv2.imshow(" Result --->>> ", tradionalOutput)
                cv2.waitKey(100)
                if output:
                    out.write(tradionalOutput)

                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                Print("Cannot read file... Exiting")
                sys.exit()

    else:

        if method.lower() == 'prnet':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            pNet = PRN(is_dlib = True)

        while(cap.isOpened()):
            count += 1

            ret, image_3 = cap.read()

            if ret == True:
                image_3 = resize_(image_3)

                if method.lower() == 'prnet':
                    pos, output = prnet_(pNet, image_3, image_3)
                    if pos is None:
                        continue
                    else:
                        print(" Frame No -->> " + str(count))
                else:
                    print("\n Traditional Approach  --->>> ")
                    # cv2.imshow("Video Frame", image_2)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    number_of_faces, all_points, hull_lists = twoFaces(image_3)
                    # print(" Total Faces-(Dest Frame) -->> ", number_of_faces)
                    if number_of_faces != 2:
                        print(" Total Faces-(Dest Frame) {} ".format(number_of_faces))
                        continue
                    else:
                        im_1_pts = all_points[0]
                        im_2_pts = all_points[1]
                        # hulls_1, hulls_2 = [], []
                        # for i in range(2):
                        #     hull_1 = hull_lists[i]
                        #     hull_2 = hull_lists[len(all_points)-1-i]
                    # hull_lists = [hulls_1, hulls_2]
                    # hull_lists = np.asarray(hull_lists)
                    hull_2 = []
                    hull_2.append(hull_lists[:2][1])
                    # print(" Both Convs: ", hull_2)
                    # print(" Both Convs: ", len(hull_lists))
                    tradionalOutput = traditionalMethods(image_3, image_3, im_1_pts, im_2_pts, hull_2, mode, save, method, visualization)
                cv2.imshow(" Result --->>> ", tradionalOutput)
                cv2.waitKey(100)
                if output:
                    out.write(out)

                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                Print("Cannot read file... Exiting")
                sys.exit()
