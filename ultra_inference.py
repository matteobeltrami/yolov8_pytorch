from ultralytics import YOLO
from yolov8 import *

if __name__ == "__main__":
    # Load model
    torch.load("yoloexport.pt")
    model = YOLO('yolov8l.pt')
    
    print("model loaded. starting inference")
    # Run inference
    results = model('dog.jpeg')
    print("done inference.")
    
    # Print image.jpg results in JSON format
    print(results[0].tojson())
