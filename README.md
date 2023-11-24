This repo contains the code to run the YOLOv8 network inference script for object detection.

To download the pretrained weights:

```[python]
python load_params.py <conf>
```
where `<conf>` is the model variant (see more [here](https://docs.ultralytics.com/tasks/detect/)) and can be `n`, `s`, `m`, `l`, `x`.

To run the prediction script:

```[python]
python yolov8.py <img_path> <conf>
```
