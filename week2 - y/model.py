from facenet_models import FacenetModel
import cv2
import numpy as np
import database as db

pic = cv2.imread('ricky_ray.jpg')

class Model:
    _model = FacenetModel()

    @staticmethod
    def recognize_faces(pic: np.ndarray, database: dict,
                        cos_dist_threshold: float = 0.5,
                        face_prob_threshold: float = 0.5) -> list:
    
        boxes, probabilities, landmarks = Model._model.detect(pic)

        # confident boxes
        valid_boxes = boxes[probabilities > face_prob_threshold] 

        # run compute_descriptors from resnet
        descriptors = Model._model.compute_descriptors(pic, valid_boxes)

        recognized_faces = []
        valid_descriptors = []

        for descriptor, box in zip(descriptors, valid_boxes):
            # match function from database
            name, distance = database.find_match(descriptor, cos_dist_threshold) 
            recognized_faces.append((name, distance, box))
            valid_descriptors.append(descriptor)
        return recognized_faces, valid_descriptors

def display_faces(pic: np.ndarray, recognized_faces: list) -> None:
    # recognized_faces is a list of tuples that contains name of the person
    # (Unknown if not found), distance between face descriptor of recognized
    # face and closest match in database, and bounding box coordinates
    # can change format later if necessary
    for name, distance, box in recognized_faces:
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        #green box if match, red box if Unknown
        rect_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) 
        pic_copy = pic.copy()
        pic_copy = cv2.rectangle(pic_copy, (x1,y1), (x2, y2), rect_color, 2)
        text = f"{name} - Distance: {distance: .2f}" if name != "Unknown" else "Unknown"
        pic_copy = cv2.putText(
          pic_copy, text, (x1, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2
        )

    cv2.imshow("Recognized Faces", pic_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()