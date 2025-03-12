import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "C:/python/faceproje/shape.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def detect_emotion(landmarks):
    
    tolerance = 2
    
    mouth_left_corner_y = landmarks[48,1]
    mouth_right_corner_y = landmarks[54,1]
    mouth_opening_center_top_y = landmarks[62,1]
    mouth_opening_center_bottom_y = landmarks[66,1]
    
    mouth_center_y = (mouth_left_corner_y + mouth_right_corner_y)/2
    mouth_opening_center_y = (mouth_opening_center_top_y + mouth_opening_center_bottom_y )/2
    
    
    if abs(mouth_center_y - mouth_opening_center_y) <= tolerance:
        
        print('neutral')
    elif mouth_center_y > mouth_opening_center_y:
        print('sad')
    elif mouth_center_y < mouth_opening_center_y:
        print('happy')  
    else:
        print('unexpected result')
    
                
image = cv2.imread('C:/python/faceproje/hapy_insan.jpg')

landmarks = get_landmarks(image)
detect_emotion(landmarks)

image_with_landmarks = annotate_landmarks(image, landmarks)

cv2.imshow('Result', image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
