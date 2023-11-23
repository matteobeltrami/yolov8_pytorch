from yolov8 import YOLOv8
import torch
import sys
import cv2
import numpy as np
from helpers import *
import torch.nn as nn

if __name__ == "__main__":
    model = YOLOv8(1, 1, 1, 80).half()
    model.load_state_dict(torch.load("./yolov8l_scratch.pt"))
    model = model.float()
    inn = torch.randn(1, 3, 64, 64)

    ultra_dict = torch.load("yolov8l.pt")
    ultra_model = ultra_dict["model"].model[:23].float()

    model.eval()
    ultra_model.eval() 

    a = [torch.randn(1, 512, 40, 40), torch.randn(1, 512, 40, 40)]
    test = ultra_dict["model"].model[11](a)
    #ultra_model = nn.Sequential(ultra_model[:9], ultra_model[9].cv1)

    f = ultra_model
    g = model.head

    # print("ULTRALYTICS:")
    # print(f)
    # print("--------")
    # print("TORCH:")
    # print(g)
    #print(f.bn.state_dict())
    #print(g.bn.state_dict())


    assert len(sys.argv) > 1 
    img_path = sys.argv[1]
    image = [cv2.imread(img_path)]
    if not isinstance(image[0], np.ndarray):
        print('Error in image loading. Check your image file.')
        sys.exit(1)
    pre_processed_image = preprocess(image)
    
    y, dt = [], []  # outputs
    x = pre_processed_image.clone()
    for i, m in enumerate(ultra_dict["model"].model):
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in ultra_dict["model"].save else None)  # save output

    z = x.clone()

    # f = ultra_dict["model"].float()
    # z_real = f(pre_processed_image)[0]

    # print(torch.isclose(z_real, z).float().mean())

    #torch
    backbone_out = model.net(pre_processed_image)
    neck_out = model.fpn(*backbone_out)
    out_torch = g(neck_out)
    # print(out_torch, "ours")
    # print(z)
    print(out_torch)
    print(z)
    print(torch.isclose(out_torch, z).float().mean())
    exit(0)

    #out_torch = g(*backbone_out)[1]

    #print(ab[0][0].shape, out_torch.shape)
    breakpoint()
    print(torch.isclose(z, out_torch).sum() / out_torch.numel())
    #print(torch.isclose(ab[0][0], out_torch[0]).sum() / out_torch[0].numel())
    #print(torch.isclose(ab[0][2], out_torch[2]).sum() / out_torch[2].numel())
    #print(torch.isclose(z, out_torch).sum() / out_torch.numel())

    import matplotlib.pyplot as plt
    ultra_result = detect_22[0]

    plt.figure(figsize=(10,6))

    plt.imshow(np.log10(abs(z[0].detach()-out_torch[0].detach())+1e-9))
    
    plt.colorbar()
    plt.show()
    # plt.imshow(out_torch[0])
    # plt.show() 
    
    #np.save("convblock/convblock_ultra.npy", out_conv_block.cpu().numpy())
    #np.save("convblock/convblock_torch.npy", out_conv_block.cpu().numpy())
