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

    ultra_dict = torch.load("../notebooks/yolov8l.pt")
    ultra_model = ultra_dict["model"].model[:23].float()

    model.eval()
    ultra_model.eval() 

    a = [torch.randn(1, 512, 40, 40), torch.randn(1, 512, 40, 40)]
    test = ultra_dict["model"].model[11](a)
    #ultra_model = nn.Sequential(ultra_model[:9], ultra_model[9].cv1)

    f = ultra_model
    g = model.head

    print("ULTRALYTICS:")
    print(f)
    print("--------")
    print("TORCH:")
    print(g)
    #print(f.bn.state_dict())
    #print(g.bn.state_dict())


    assert len(sys.argv) > 1 
    img_path = sys.argv[1]
    image = [cv2.imread(img_path)]
    if not isinstance(image[0], np.ndarray):
        print('Error in image loading. Check your image file.')
        sys.exit(1)
    pre_processed_image = preprocess(image)

    ultra_outputs = [pre_processed_image]
    for i in range(10):
        ultra_outputs.append(ultra_dict["model"].model[i](ultra_outputs[-1]))
    
    
    #ultra 11
    upsample_10 = ultra_dict["model"].model[10](ultra_outputs[-1])
    concat_11 = ultra_dict["model"].model[11]([upsample_10, ultra_outputs[7]])
    ultra_outputs.append(upsample_10)
    ultra_outputs.append(concat_11)

    #ultra 12
    c2f_12_cv1 = ultra_dict["model"].model[12].cv1(ultra_outputs[-1])
    y = torch.chunk(c2f_12_cv1, chunks=2, dim=1)
    c2f_12_m0 = ultra_dict["model"].model[12].m[0](y[1])
    c2f_12_m1 = ultra_dict["model"].model[12].m[1](c2f_12_m0)
    c2f_12_m2 = ultra_dict["model"].model[12].m[2](c2f_12_m1)
    cat = torch.cat((y[0], y[1], c2f_12_m0, c2f_12_m1, c2f_12_m2), dim=1)
    c2f_12 = ultra_dict["model"].model[12].cv2(cat)
    ultra_outputs.append(c2f_12)

    #ultra 13
    upsample_13 = ultra_dict["model"].model[13](ultra_outputs[-1])
    ultra_outputs.append(upsample_13)

    #ultra 14
    concat_14 = ultra_dict["model"].model[14]([upsample_13, ultra_outputs[5]])
    ultra_outputs.append(concat_14)

    #ultra 15
    c2f_15_cv1 = ultra_dict["model"].model[15].cv1(ultra_outputs[-1])
    y = torch.chunk(c2f_15_cv1, chunks=2, dim=1)
    c2f_15_m0 = ultra_dict["model"].model[15].m[0](y[1])
    c2f_15_m1 = ultra_dict["model"].model[15].m[1](c2f_15_m0)
    c2f_15_m2 = ultra_dict["model"].model[15].m[2](c2f_15_m1)
    cat = torch.cat((y[0], y[1], c2f_15_m0, c2f_15_m1, c2f_15_m2), dim=1)
    c2f_15 = ultra_dict["model"].model[15].cv2(cat)
    ultra_outputs.append(c2f_15)

    #ultra 16
    conv_16 = ultra_dict["model"].model[16](ultra_outputs[-1])
    ultra_outputs.append(conv_16)

    #ultra 17
    concat_17 = ultra_dict["model"].model[17]([conv_16, ultra_outputs[13]])
    ultra_outputs.append(concat_17)

    #ultra 18
    c2f_18_cv1 = ultra_dict["model"].model[18].cv1(ultra_outputs[-1])
    y = torch.chunk(c2f_18_cv1, chunks=2, dim=1)
    c2f_18_m0 = ultra_dict["model"].model[18].m[0](y[1])
    c2f_18_m1 = ultra_dict["model"].model[18].m[1](c2f_18_m0)
    c2f_18_m2 = ultra_dict["model"].model[18].m[2](c2f_18_m1)
    cat = torch.cat((y[0], y[1], c2f_18_m0, c2f_18_m1, c2f_18_m2), dim=1)
    c2f_18 = ultra_dict["model"].model[18].cv2(cat)
    ultra_outputs.append(c2f_18)

    #ultra 19
    conv_19 = ultra_dict["model"].model[19](ultra_outputs[-1])
    ultra_outputs.append(conv_19)

    #ultra 20
    concat_20 = ultra_dict["model"].model[20]([conv_19, ultra_outputs[10]])
    ultra_outputs.append(concat_20)

    #ultra 21
    c2f_21_cv1 = ultra_dict["model"].model[21].cv1(ultra_outputs[-1])
    y = torch.chunk(c2f_21_cv1, chunks=2, dim=1)
    c2f_21_m0 = ultra_dict["model"].model[21].m[0](y[1])
    c2f_21_m1 = ultra_dict["model"].model[21].m[1](c2f_21_m0)
    c2f_21_m2 = ultra_dict["model"].model[21].m[2](c2f_21_m1)
    cat = torch.cat((y[0], y[1], c2f_21_m0, c2f_21_m1, c2f_21_m2), dim=1)
    c2f_21 = ultra_dict["model"].model[21].cv2(cat)
    ultra_outputs.append(c2f_21)

    #detect 22
    #breakpoint()
    detect_22 = ultra_dict["model"].model[22]([c2f_15, c2f_18, c2f_21])

    from yolov8 import autopad
    from yolov8 import make_anchors
    from yolov8 import dist2bbox
    from yolov8 import DFL

    x=[c2f_15, c2f_18, c2f_21]
    ab=[]
    for i in range(len((256, 512, 512))):
        a = ultra_dict["model"].model[22].cv2[i](x[i])
        b = ultra_dict["model"].model[22].cv3[i](x[i])
        ab.append((a, b, x[i]))
        x[i] = torch.cat((a, b), dim=1)
    anchors, strides = (
        ii.transpose(0, 1) for ii in make_anchors(x, [8, 16, 32], 0.5)
    )
    y = [(i.reshape(x[0].shape[0], 144, -1)) for i in x]
    x_cat = torch.cat((y[0], y[1], y[2]), dim=2)
    box, cls = x_cat[:, : 16 * 4], x_cat[:, 16 * 4 :]
    dbox = (
        dist2bbox(DFL(16)(box), anchors.unsqueeze(0), xywh=True, dim=1)
        * strides
    )
    z = torch.cat((dbox, nn.Sigmoid()(cls)), dim=1)

    torch.save(z, "ultra_model.pt")
    

    #torch
    backbone_out = model.net(pre_processed_image)
    neck_out = model.fpn(*backbone_out)
    #out_torch = g(*backbone_out)[1]
    out_torch = g(neck_out)#[2]c

    #print(ab[0][0].shape, out_torch.shape)
    breakpoint()
    print(torch.isclose(z, out_torch).sum() / out_torch.numel())
    #print(torch.isclose(ab[0][0], out_torch[0]).sum() / out_torch[0].numel())
    #print(torch.isclose(ab[0][2], out_torch[2]).sum() / out_torch[2].numel())
    #print(torch.isclose(z, out_torch).sum() / out_torch.numel())
    breakpoint()

    import matplotlib.pyplot as plt
    ultra_result = detect_22[0]

    breakpoint()
    plt.figure(figsize=(10,6))

    print(np.log10(abs(ultra_result[0].detach()-out_torch[0].detach())+1e-9))
    plt.imshow(np.log10(abs(z[0].detach()-out_torch[0].detach())+1e-9))
    
    plt.colorbar()
    plt.show()
    # plt.imshow(out_torch[0])
    # plt.show() 

    print(torch.isclose(ultra_result, out_torch).sum() / out_torch.numel())
    
    #np.save("convblock/convblock_ultra.npy", out_conv_block.cpu().numpy())
    #np.save("convblock/convblock_torch.npy", out_conv_block.cpu().numpy())