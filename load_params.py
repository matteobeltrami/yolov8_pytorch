import numpy as np
import torch
import sys

from yolov8 import YOLOv8


def count_id(x, key): return len([x_ for x_ in x if key in x_])

if __name__ == "__main__":
    model = YOLOv8(1,1,1)
    ii = list(model.state_dict().keys())
    our_dict = model.state_dict()
    dat = np.load("yolov8l.npz")
    uu = dat.files
    oo = list(torch.load("yolov8l.pt")["model"].state_dict())

    if len(sys.argv) > 1:
        key = str(sys.argv[1])
        print("Counting elements with key %s." % key)
        print(count_id(ii, key))
        print(count_id(oo, key))
        print(count_id(uu, key))

    for i,u in enumerate(uu):
        assert our_dict[ii[i]].shape == dat[u].shape
        our_dict[ii[i]] = torch.Tensor(dat[u])

    torch.save(our_dict, "yolov8l_scratch.pt")

