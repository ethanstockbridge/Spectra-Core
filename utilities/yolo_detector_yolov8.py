from math import e

import cv2
from numpy import asarray, uint8
from PIL import Image

from config.config_manager import ConfigManager
from ultralytics import YOLO

#TODO: TBD: Make yolo detector a separate service?

collision_thresh=0.5

class RectObject:
    """Rectangle that determines the location of an object
    """
    def __init__(self, x, y, width, height):
        """Create a RectObject

        Args:
            x (int): X
            y (int): Y
            width (int): Width
            height (int): Height
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def collides(self, other):
        """Check for collision between another object and this object with a given threshold

        Args:
            other (RectObject): Other

        Returns:
            bool: Whether they collide with a given threshold
        """
        x_overlap_area = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y_overlap_area = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))

        self_area = self.width * self.height
        other_area = other.width * other.height
        overlap_area = x_overlap_area * y_overlap_area

        self_overlap_ratio = overlap_area / self_area
        other_overlap_ratio = overlap_area / other_area

        overlapped = self_overlap_ratio > collision_thresh and other_overlap_ratio > collision_thresh

        return overlapped


class Board:
    """Virtual board that holds all detections
    """
    def __init__(self):
        self.objects = []

    def add_object(self, obj):
        """Add object to board

        Args:
            obj (RectObject): Object
        """
        self.objects.append(obj)

    def collides(self, new_obj):
        """Check for collision

        Args:
            new_obj (RectObject): Object

        Returns:
            bool: If collides with any objects?
        """
        for obj in self.objects:
            if obj.collides(new_obj):
                return True
        return False


class YoloDetector():
    """Handle yolo loading and detection
    """
    def __init__(self):
        self.__config = ConfigManager()
        try:
            self.model = YOLO(self.__config["paths"]["model_path"])
        except:
            print("ERROR: model not found, change model path in config file")
            exit(1)

    def predict(self, input_image_path):
        """Predict the detections on an image

        Args:
            input_image_path (str): Image file name

        Returns:
            Array: Yolo results
        """
        results = self.model(input_image_path)  # inference
        return results
    
    def save_labels(self, output_path):
        text_results = ""
        for name in self.model.names.values():
            text_results+=name+"\n"
        with open(output_path, "w") as file:
            file.write(text_results)

    def plot_predict(self, input_image_path, output_image_path, min_conf):
        """Save predictions to a new file showing bounding boxes

        Args:
            input_image_path (str): Input image path
            output_image_path (str): Output image path
            min_conf (float): Min confidence to trigger
        """
        results = self.predict(input_image_path)

        boxes = results[0].boxes

        image = Image.open(input_image_path)
        frame = asarray(image, dtype=uint8)
        text_results = ""
        image_size = self.__config["yolo"]["size"]
        board = Board()
        for i in range(0,len(boxes)):
            box = boxes[i]
            conf = box.conf.tolist()[0]
            xyxy = box.xyxy.tolist()[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            label = int(box.cls.tolist()[0])
            label_name = self.model.names[box.cls.tolist()[0]]
            if conf>min_conf: #not high enough confidence
                det = RectObject(x1,y1,x2-x1,y2-y1)
                if not board.collides(det):
                    board.add_object(det)
                    # yolo format: center_x, center_y, width, height
                    cx,cy,w,h = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2, xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                    text_results += f"{label} {cx/image_size} {cy/image_size} {w/image_size} {h/image_size}\n"
                    # cv format: x1y1 x2y2
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                    cv2.putText(frame, label_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                else:
                    print("Object collided")
        Image.fromarray(frame).save(output_image_path)
        with open(input_image_path.replace(self.__config["spectrogram"]["format"],"txt"), "w") as file:
            file.write(text_results)