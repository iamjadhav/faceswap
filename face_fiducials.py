import cv2
import dlib
import numpy as np
from imutils import face_utils as utils

def face_fiducials(frame):

    imp_pts = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = hog_face_detector(gray, 1)
    total_faces = len(faces)
    face_landmarks_list = []
    hull_list = []
    # print("Total Faces in current Frame:-> ", total_faces)
    for i,face in enumerate(faces):

        face_landmarks = dlib_facelandmark(gray, face)
        face_landmarks = utils.shape_to_np(face_landmarks)
        face_landmarks_list.append(face_landmarks)
        # print(" Face LandMarks ", len(face_landmarks))
        (p, q, w, h) = utils.rect_to_bb(face)
        cv2.rectangle(frame, (p, q), (p + w, q + h), (0,255,255), 1)

        # if visualization == True:
        for (p,q) in face_landmarks:
            cv2.circle(frame, (p, q), 2, (0, 0, 255), -1)
            imp_pts.append((p,q))

        hull = cv2.convexHull(np.array(imp_pts), False)
        hull = np.reshape(hull, (hull.shape[0], hull.shape[2]))
        hull_list.append(hull)
        # print(" Hull List ", hull_list)

        # cv2.imshow("Face Landmarks", frame)
        # print(len(imp_pts))

    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break
    #
    # cv2.destroyAllWindows()

    return total_faces, imp_pts, hull_list


def twoFaces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rects = detector(gray,1)

    points = []
    faces = []
    hull_list = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num_faces = len(rects)

    if(num_faces==2):
        for (i,rect) in enumerate(rects):

            shape = predictor(gray,rect)
            shape = utils.shape_to_np(shape)
            (x,y,w,h) = utils.rect_to_bb(rect)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            for (x,y) in shape:
                cv2.circle(img,(x,y),2,(0,0,255),-1)
                points.append((x,y))

            hull = cv2.convexHull(np.array(points), False)
            hull = np.reshape(hull, (hull.shape[0], hull.shape[2]))
            hull_list.append(hull)

            faces.append(points)
            points = []

    return num_faces, faces, hull_list
