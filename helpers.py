# thanks tinygrad
import numpy as np
from itertools import chain
from pathlib import Path
import cv2
from collections import defaultdict
import time, io, sys
import torch


# Model architecture from https://github.com/ultralytics/ultralytics/issues/189
# The upsampling class has been taken from this pull request https://github.com/tinygrad/tinygrad/pull/784 by dc-dc-dc. Now 2(?) models use upsampling. (retinet and this)


# Pre processing image functions.
def compute_transform(
    image, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
):
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0) if not scaleup else r
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = (np.mod(dw, stride), np.mod(dh, stride)) if auto else (0.0, 0.0)
    new_unpad = (new_shape[1], new_shape[0]) if scaleFill else new_unpad
    dw /= 2
    dh /= 2
    image = (
        cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        if shape[::-1] != new_unpad
        else image
    )
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image


def preprocess(im, imgsz=640, model_stride=32, model_pt=True):
    same_shapes = all(x.shape == im[0].shape for x in im)
    auto = same_shapes and model_pt
    im = torch.Tensor(
        [
            compute_transform(x, new_shape=imgsz, auto=auto, stride=model_stride)
            for x in im
        ]
    )
    im = torch.stack(im) if im.shape[0] > 1 else im
    # im = im[..., ::-1].permute(0, 3, 1, 2)  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im = torch.flip(im, (-1,)).permute(0, 3, 1, 2)  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

# Post Processing functions
def box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def box_iou(box1, box2):
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = box_area(box1)[:, None]
    area2 = box_area(box2)[None, :]
    iou = inter / (area1 + area2 - inter)
    return iou


def compute_nms(boxes, scores, iou_threshold):
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou = box_iou(boxes[i][None, :], boxes[order[1:]])
        inds = np.where(iou.squeeze() <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    agnostic=False,
    max_det=300,
    nc=0,
    max_wh=7680,
):
    prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    bs, nc = prediction.shape[0], nc or (prediction.shape[1] - 4)
    xc = np.amax(prediction[:, 4 : 4 + nc], axis=1) > conf_thres
    nm = prediction.shape[1] - nc - 4
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):
        x = x.swapaxes(0, -1)[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls, mask = np.split(x, [4, 4 + nc], axis=1)
        conf, j = np.max(cls, axis=1, keepdims=True), np.argmax(
            cls, axis=1, keepdims=True
        )
        x = np.concatenate((xywh2xyxy(box), conf, j.astype(np.float32), mask), axis=1)
        x = x[conf.ravel() > conf_thres]
        if not x.shape[0]:
            continue
        x = x[np.argsort(-x[:, 4])]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = compute_nms(boxes, scores, iou_thres)[:max_det]
        output[xi] = x[i]
    return output


def postprocess(preds, img, orig_imgs):
    print("copying to CPU now for post processing")
    # if you are on CPU, this causes an overflow runtime error. doesn't "seem" to make any difference in the predictions though.
    # TODO: make non_max_suppression in tinygrad - to make this faster
    preds = preds.numpy()
    preds = non_max_suppression(
        prediction=preds, conf_thres=0.25, iou_thres=0.5, agnostic=True, max_det=300
    )
    all_preds = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        if not isinstance(orig_imgs, torch.Tensor):
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            all_preds.append(pred)
    return all_preds


def draw_bounding_boxes_and_save(
    orig_img_paths, output_img_paths, all_predictions, class_labels, iou_threshold=0.5
):
    color_dict = {
        label: tuple(
            (((i + 1) * 50) % 256, ((i + 1) * 100) % 256, ((i + 1) * 150) % 256)
        )
        for i, label in enumerate(class_labels)
    }
    font = cv2.FONT_HERSHEY_SIMPLEX

    def is_bright_color(color):
        r, g, b = color
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return brightness > 127

    for img_idx, (orig_img_path, output_img_path, predictions) in enumerate(
        zip(orig_img_paths, output_img_paths, all_predictions)
    ):
        predictions = np.array(predictions)
        orig_img = cv2.imread(orig_img_path)
        height, width, _ = orig_img.shape
        box_thickness = int((height + width) / 400)
        font_scale = (height + width) / 2500

        grouped_preds = defaultdict(list)
        object_count = defaultdict(int)

        for pred_np in predictions:
            grouped_preds[int(pred_np[-1])].append(pred_np)

        def draw_box_and_label(pred, color):
            x1, y1, x2, y2, conf, _ = pred
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, box_thickness)
            label = f"{class_labels[class_id]} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
            label_y, bg_y = (
                (y1 - 4, y1 - text_size[1] - 4)
                if y1 - text_size[1] - 4 > 0
                else (y1 + text_size[1], y1)
            )
            cv2.rectangle(
                orig_img,
                (x1, bg_y),
                (x1 + text_size[0], bg_y + text_size[1]),
                color,
                -1,
            )
            font_color = (0, 0, 0) if is_bright_color(color) else (255, 255, 255)
            cv2.putText(
                orig_img,
                label,
                (x1, label_y),
                font,
                font_scale,
                font_color,
                1,
                cv2.LINE_AA,
            )

        for class_id, pred_list in grouped_preds.items():
            pred_list = np.array(pred_list)
            while len(pred_list) > 0:
                max_conf_idx = np.argmax(pred_list[:, 4])
                max_conf_pred = pred_list[max_conf_idx]
                pred_list = np.delete(pred_list, max_conf_idx, axis=0)
                color = color_dict[class_labels[class_id]]
                draw_box_and_label(max_conf_pred, color)
                object_count[class_labels[class_id]] += 1
                iou_scores = box_iou(np.array([max_conf_pred[:4]]), pred_list[:, :4])
                low_iou_indices = np.where(iou_scores[0] < iou_threshold)[0]
                pred_list = pred_list[low_iou_indices]
                for low_conf_pred in pred_list:
                    draw_box_and_label(low_conf_pred, color)

        print(f"Image {img_idx + 1}:")
        print("Objects detected:")
        for obj, count in object_count.items():
            print(f"- {obj}: {count}")

        cv2.imwrite(output_img_path, orig_img)
        print(f"saved detections at {output_img_path}")


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])  # y1, y2
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    gain = (
        ratio_pad
        if ratio_pad
        else min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    )
    pad = (
        (img1_shape[1] - img0_shape[1] * gain) / 2,
        (img1_shape[0] - img0_shape[0] * gain) / 2,
    )
    boxes_np = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    boxes_np[..., [0, 2]] -= pad[0]
    boxes_np[..., [1, 3]] -= pad[1]
    boxes_np[..., :4] /= gain
    boxes_np = clip_boxes(boxes_np, img0_shape)
    return boxes_np


def xywh2xyxy(x):
    xy = x[..., :2]  # center x, y
    wh = x[..., 2:4]  # width, height
    xy1 = xy - wh / 2  # top left x, y
    xy2 = xy + wh / 2  # bottom right x, y
    result = np.concatenate((xy1, xy2), axis=-1)
    return torch.Tensor(result)


def label_predictions(all_predictions):
    class_index_count = defaultdict(int)
    for predictions in all_predictions:
        predictions = np.array(predictions)
        for pred_np in predictions:
            class_id = int(pred_np[-1])
            class_index_count[class_id] += 1

    return dict(class_index_count)
