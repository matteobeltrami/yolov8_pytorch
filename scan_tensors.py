import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = Path("backbone")

    input_torch = np.load(folder.joinpath("input_torch.npy"))
    input_tinygrad = np.load(folder.joinpath("input_tinygrad.npy"))
    breakpoint()
    print("input", f"{1 - np.isclose(input_torch, input_tinygrad).sum() / input_torch.size:.3f}")
    print("-----")
    for i, block in enumerate([f"b{i}" for i in range(1, 6)]):
        torch = np.load(folder.joinpath(block + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(block + "_tinygrad.npy"))

        # if i+1 in [2, 3, 5]:
        print(block, f"{1 - np.isclose(torch, tinygrad).sum() / torch.size:.3f}")

    folder = Path("convblock")
    print("----- " + str(folder.name))
    for layer in ["conv", "bn", "silu"]:
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))

        # print(layer, f"{1 - np.isclose(torch, tinygrad).sum() / torch.size:.3f}")
        print(layer, f"{1 - np.isclose(torch, tinygrad, rtol=1e-05, atol=2e-08).sum() / torch.size:.3f}")

    folder = Path("norm")
    print("----- " + str(folder.name))
    for layer in ["bn_ones", "conv_ones"]:
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))

        # print(layer, f"{1 - np.isclose(torch, tinygrad).sum() / torch.size:.3f}")
        print(layer, f"{1 - np.isclose(torch, tinygrad, rtol=1e-05, atol=2e-08).sum() / torch.size:.3f}")

    folder = Path("network")
    print("----- " + str(folder.name))
    for i, layer in enumerate(["net0", "net1", "net2", "fpn0", "fpn1", "fpn2"]):
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))

        # print(layer, f"{1 - np.isclose(torch, tinygrad).sum() / torch.size:.3f}")
        print(layer, f"{1 - np.isclose(torch, tinygrad, rtol=1e-05, atol=2e-08).sum() / torch.size:.3f}")

    folder = Path("sanity_check")
    print("----- " + str(folder.name))
    for layer in ["conv_ones"]:
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))
        ultra = np.load(folder.joinpath(layer + "_ultra.npy"))

        print(layer, "between torch and tinygrad ", f"{1 - np.isclose(torch, tinygrad, rtol=1e-05, atol=1e-08).sum() / torch.size:.3f}")
        print(layer, "between torch and ultralytics ", f"{1 - np.isclose(torch, ultra, rtol=1e-05, atol=1e-08).sum() / torch.size:.3f}")
        tmp = np.sum(np.squeeze(np.isclose(torch, tinygrad)), axis=0) 
        plt.imshow(tmp, cmap='binary')
        plt.axis('off')
        plt.colorbar()
        plt.savefig(f'sanity_check/{layer}.png', dpi=300, bbox_inches='tight')

    folder = Path("conv1")
    print("----- " + str(folder.name))
    for layer in ["w"]:
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))

        # print(layer, f"{1 - np.isclose(torch, tinygrad).sum() / torch.size:.3f}")
        print(layer, f"{1 - np.isclose(torch, tinygrad, rtol=1e-05, atol=2e-08).sum() / torch.size:.3f}")

        tmp = np.sum(np.squeeze(np.isclose(torch, tinygrad)), axis=0) 
        # plt.imshow(tmp, cmap='binary')
        # plt.axis('off')
        # plt.colorbar()
        # plt.savefig(f'sanity_check/{layer}.png', dpi=300, bbox_inches='tight')
