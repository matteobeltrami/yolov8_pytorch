import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *
from pathlib import Path
import numpy as np
import sys
import cv2


# this function is from the original implementation
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(h, dtype=torch.float32) + grid_cell_offset

        grid_x, grid_y = torch.meshgrid(sx, sy, indexing="ij")  # to avoid warning

        anchor_points.append(torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w,), stride, dtype=torch.float32))

    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0).unsqueeze(1)

    return anchor_points, stride_tensor


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.chunk(distance, chunks=2, dim=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=1)
    return torch.cat((x1y1, x2y2), dim=1)


class Upsample:
    def __init__(self, scale_factor, mode="nearest"):
        assert mode == "nearest"
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: torch.Tensor):
        assert (
            len(x.shape) > 2 and len(x.shape) <= 5
        ), "Input tensor must have 3 to 5 dimensions"
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return upsampled


class Conv_Block(nn.Module):
    def __init__(
        self, c1, c2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2, eps=0.001)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        np.save("convblock/conv_torch.npy", x.detach().numpy())
        x = self.bn(x)
        np.save("convblock/bn_torch.npy", x.detach().numpy())
        x = self.silu(x)
        np.save("convblock/silu_torch.npy", x.detach().numpy())
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        shortcut: bool,
        groups=1,
        kernels: list = (3, 3),
        channel_factor=0.5,
    ):
        super().__init__()
        c_ = int(c2 * channel_factor)
        self.cv1 = Conv_Block(c1, c_, kernel_size=kernels[0], stride=1, padding=1)
        self.cv2 = Conv_Block(
            c_, c2, kernel_size=kernels[1], stride=1, padding=1, groups=groups
        )
        self.residual = c1 == c2 and shortcut

    def forward(self, x):
        if self.residual:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, groups=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv_Block(
            c1,
            2 * self.c,
            1,
        )
        self.cv2 = Conv_Block((2 + n) * self.c, c2, 1)
        self.bottleneck = nn.ModuleList([
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                groups,
                kernels=[(3, 3), (3, 3)],
                channel_factor=1.0,
            )
            for _ in range(n)
        ])

    def forward(self, x):
        x = self.cv1(x)
        y = list(torch.chunk(x, chunks=2, dim=1))
        y.extend(m(y[-1]) for m in self.bottleneck)
        z = y[0]
        for i in y[1:]:
            z = torch.cat((z, i), dim=1)
        return self.cv2(z)


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_Block(c1, c_, 1, 1, padding=0)
        self.cv2 = Conv_Block(c_ * 4, c2, 1, 1, padding=0)
        self.maxpool = lambda x: F.max_pool2d(
            F.pad(x, (k // 2, k // 2, k // 2, k // 2)), kernel_size=k, stride=1
        )

    def forward(self, x):
        x = self.cv1(x)
        x2 = self.maxpool(x)
        x3 = self.maxpool(x2)
        x4 = self.maxpool(x3)

        y = torch.cat((x, x2, x3, x4), dim=1)
        return self.cv2(y)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        weight = torch.arange(c1).reshape(1, c1, 1, 1).float()
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(weight)
        self.c1 = c1

    @torch.no_grad() # TODO: check when training
    def forward(self, x):
        b, c, a = x.shape
        y = x.reshape(b, 4, self.c1, a).transpose(2, 1)
        y = F.softmax(y, dim=1)
        y = self.conv(y)
        y = y.reshape(b, 4, a)
        return y


class Darknet(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.b1 = nn.Sequential(
            Conv_Block(c1=3, c2=int(64 * w), kernel_size=3, stride=2, padding=1),
            Conv_Block(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            C2f(c1=int(128 * w), c2=int(128 * w), n=round(3 * d), shortcut=True),
            Conv_Block(int(128 * w), int(256 * w), 3, 2, 1),
            C2f(int(256 * w), int(256 * w), round(6 * d), True),
        )
        self.b3 = nn.Sequential(
            Conv_Block(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1),
            C2f(int(512 * w), int(512 * w), round(6 * d), True),
        )
        self.b4 = nn.Sequential(
            Conv_Block(
                int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1
            ),
            C2f(int(512 * w * r), int(512 * w * r), round(3 * d), True),
        )

        self.b5 = SPPF(int(512 * w * r), int(512 * w * r), 5)

    def return_modules(self):
        return [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        np.save("backbone/input_torch.npy", x.detach().numpy())
        x1 = self.b1(x)
        np.save("backbone/b1_torch.npy", x1.detach().numpy())
        x2 = self.b2(x1)
        np.save("backbone/b2_torch.npy", x2.detach().numpy())
        x3 = self.b3(x2)
        np.save("backbone/b3_torch.npy", x3.detach().numpy())
        x4 = self.b4(x3)
        np.save("backbone/b4_torch.npy", x4.detach().numpy())
        x5 = self.b5(x4)
        np.save("backbone/b5_torch.npy", x5.detach().numpy())
        return (x2, x3, x5)


class Yolov8Neck(nn.Module):
    def __init__(self, w, r, d):
        super().__init__()
        self.up = Upsample(2, mode="nearest")
        self.n1 = C2f(
            c1=int(512 * w * (1 + r)), c2=int(512 * w), n=round(3 * d), shortcut=False
        )
        self.n2 = C2f(c1=int(768 * w), c2=int(256 * w), n=round(3 * d), shortcut=False)
        self.n3 = Conv_Block(
            c1=int(256 * w), c2=int(256 * w), kernel_size=3, stride=2, padding=1
        )
        self.n4 = C2f(c1=int(768 * w), c2=int(512 * w), n=round(3 * d), shortcut=False)
        self.n5 = Conv_Block(
            c1=int(512 * w), c2=int(512 * w), kernel_size=3, stride=2, padding=1
        )
        self.n6 = C2f(
            c1=int(512 * w * (1 + r)),
            c2=int(512 * w * r),
            n=round(3 * d),
            shortcut=False,
        )

    def return_modules(self):
        return [self.n1, self.n2, self.n3, self.n4, self.n5, self.n6]

    def forward(self, p3, p4, p5):
        x = self.up(p5)
        x = torch.cat((x, p4), dim=1)
        x = self.n1(x)
        h1 = self.up(x)
        h1 = torch.cat((h1, p3), dim=1)
        head_1 = self.n2(h1)
        h2 = self.n3(head_1)
        h2 = torch.cat((h2, x), dim=1)
        head_2 = self.n4(h2)
        h3 = self.n5(head_2)
        h3 = torch.cat((h3, p5), dim=1)
        head_3 = self.n6(h3)
        return [head_1, head_2, head_3]


class DetectionHead(nn.Module):
    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16
        self.nc = nc
        self.nl = len(filters)
        self.no = nc + self.ch * 4
        self.stride = [8, 16, 32]
        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))
        self.cv2 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv_Block(x, c2, 3, padding=1),
                    Conv_Block(c2, c2, 3, padding=1),
                    nn.Conv2d(c2, 4 * self.ch, 1),
                )
                for x in filters
            ]
        )
        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    Conv_Block(x, c1, 3, padding=1),
                    Conv_Block(c1, c1, 3, padding=1),
                    nn.Conv2d(c1, self.nc, 1),
                )
                for x in filters
            ]
        )
        self.dfl = DFL(self.ch)

    def forward(self, x):
        for i in range(self.nl):
            a = self.cv2[i](x[i])
            b = self.cv3[i](x[i])
            x[i] = torch.cat((a, b), dim=1)
        self.anchors, self.strides = (
            ii.transpose(0, 1) for ii in make_anchors(x, self.stride, 0.5)
        )
        y = [(i.reshape(x[0].shape[0], self.no, -1)) for i in x]
        x_cat = torch.cat((y[0], y[1], y[2]), dim=2)
        box, cls = x_cat[:, : self.ch * 4], x_cat[:, self.ch * 4 :]
        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        z = torch.cat((dbox, nn.Sigmoid()(cls)), dim=1)
        return z


class YOLOv8(nn.Module):
    def __init__(self, w, r, d, num_classes=80):
        super().__init__()
        self.net = Darknet(w, r, d)
        self.fpn = Yolov8Neck(w, r, d)
        self.head = DetectionHead(
            num_classes, filters=(int(256 * w), int(512 * w), int(512 * w * r))
        )

    def forward(self, x):
        x = self.net(x)
        np.save("network/net0_torch.npy", x[0].numpy())
        np.save("network/net1_torch.npy", x[1].numpy())
        np.save("network/net2_torch.npy", x[2].numpy())
        x = self.fpn(*x)
        np.save("network/fpn0_torch.npy", x[0].numpy())
        np.save("network/fpn1_torch.npy", x[1].numpy())
        np.save("network/fpn2_torch.npy", x[2].numpy())
        x = self.head(x)
        return self.head(x)

    def return_all_trainable_modules(self):
        backbone_modules = [*range(10)]
        yolov8neck_modules = [12, 15, 16, 18, 19, 21]
        yolov8_head_weights = [(22, self.head)]
        return [
            *zip(backbone_modules, self.net.return_modules()),
            *zip(yolov8neck_modules, self.fpn.return_modules()),
            *yolov8_head_weights,
        ]


def get_variant_multiples(variant):
    tmp = {
        "n": (0.33, 0.25, 2.0),
        "s": (0.33, 0.50, 2.0),
        "m": (0.67, 0.75, 1.5),
        "l": (1.0, 1.0, 1.0),
        "x": (1, 1.25, 1.0),
    }.get(variant, None)

    return tmp[1], tmp[2], tmp[0]


if __name__ == "__main__":
    assert len(sys.argv) > 1
    print(len(sys.argv))
    if len(sys.argv) <= 2:
        print("Falling back on L model. Configuration was not passed.")
        conf = "l"
    else:
        conf = str(sys.argv[2])

    output_folder_path = Path('./outputs_yolov8')
    output_folder_path.mkdir(parents=True, exist_ok=True)

    img_paths = [sys.argv[1]]
    for img_path in img_paths:
        image = [cv2.imread(img_path)]
        out_paths = [(output_folder_path / f"{Path(img_path).stem}_output{Path(img_path).suffix}").as_posix()]
        if not isinstance(image[0], np.ndarray):
            print('Error in image loading. Check your image file.')
            sys.exit(1)
        pre_processed_image = preprocess(image)
    
        model = YOLOv8(*get_variant_multiples(conf), num_classes=80)
        model.load_state_dict(torch.load("yolov8l_scratch.pt"))
        model.eval()
    
        import torchvision
        torchvision.utils.save_image(pre_processed_image, "blabla.png")
        st = time.time()
        with torch.no_grad():
            predictions = model(pre_processed_image)
        print(f'did inference in {int(round(((time.time() - st) * 1000)))}ms')
    
        with torch.no_grad():
            post_predictions = postprocess(preds=predictions, img=pre_processed_image, orig_imgs=image)
    
        class_labels = [s.strip() for s in open("coco.names", "r").readlines()]
    
        draw_bounding_boxes_and_save(orig_img_paths=img_paths, output_img_paths=out_paths, all_predictions=post_predictions, class_labels=class_labels)
