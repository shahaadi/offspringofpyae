from facenet_models import FacenetModel
import cv2
import numpy as np
import database as db

pic = cv2.imread('ricky_ray.jpg')

class Model:
    def __init__(self):
        self._model = FacenetModel()

    def recognize_faces(self, pic: np.ndarray, database: dict,
                        cos_dist_threshold: float = 0.5,
                        face_prob_threshold: float = 0.5) -> list:
    
        boxes, probabilities, landmarks = self._model.detect(pic)

        # confident boxes
        valid_boxes = boxes[probabilities > face_prob_threshold] 

        # run compute_descriptors from resnet
        descriptors = self._model.compute_descriptors(pic, valid_boxes)

        recognized_faces = []

        for descriptor, box in zip(descriptors, valid_boxes):
            # match function from database
            name, distance = db.find_match(descriptor, database, cos_dist_threshold) 
            recognized_faces.append((name, distance, box))
        return recognized_faces

def display_faces(pic: np.ndarray, recognized_faces: list) -> None:
    # recognized_faces is a list of tuples that contains name of the person
    # (Unknown if not found), distance between face descriptor of recognized
    # face and closest match in database, and bounding box coordinates
    # can change format later if necessary
    for name, distance, box in recognized_faces:
        x, y, w, h = box
        #green box if match, red box if Unknown
        rect_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
        pic = cv2.rectangle(pic, (x,y), (x+w, y+h), rect_color, 2)
        text = f"{name} - Distance: {distance: .2f}" if name != "Unknown" else "Unknown"
        pic = cv2.putText(
          pic, text, (x, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2
        )

    cv2.imshow("Recognized Faces", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()