import numpy as np
import cv2
import utils
import dlib
from imutils import face_utils




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Face Swapping')

    parser.add_argument('--source_frame', default='TestData/Rambo.jpg', type=str, help='Path to the Source Frame e.g Data/Avril.jpg OR TestData/Rambo.jpg')
    parser.add_argument('--dst_frame', default='TestData/Test1.mp4', type=str, help='Path to the Destination Video Frame e.g Data/Data1.mp4 OR TestData/Test1.mp4')
    parser.add_argument('--method', default='tps', type=str, help='Warping Approach to use e.g TPS OR Delaunay Triangulation OR PRNet')
    parser.add_argument('--mode', default='1', type=str, help='Mode of Swapping e.g 1 for Swapping Source Frame with a Destination Image OR 2 for Swapping Source Image with Destination Video Frame')
    parser.add_argument('--draw', default=True, type=bool, help='Triangle Visualization for frames')
    parser.add_argument('--output', default=False, type=bool, help='To save the Output or not')
    parser.add_argument('--resize', default=False, type=bool, help='To resize the Input Frames or not')

    arguments = parser.parse_args()

    src_dir = args.source_frame
    dst_dir = args.dst_frame
    method = args.method
    mode = args.mode
    visualization = args.draw
    save = args.output
    resize = args.resize
