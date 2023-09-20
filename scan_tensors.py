import numpy as np
from pathlib import Path

if __name__ == "__main__":
    folder = Path("backbone")

    input_torch = np.load(folder.joinpath("input_torch.npy"))
    input_tinygrad = np.load(folder.joinpath("input_tinygrad.npy"))
    # print("input", np.isclose(input_torch, input_tinygrad).sum() / input_torch.size)
    for i, block in enumerate([f"b{i}" for i in range(1, 6)]):
        torch = np.load(folder.joinpath(block + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(block + "_tinygrad.npy"))

        if i+1 in [2, 3, 5]:
            print(block, np.isclose(torch, tinygrad).sum() / torch.size)

    # folder = Path("convblock")
    # for layer in ["conv", "bn", "silu"]:
        # torch = np.load(folder.joinpath(layer + "_torch.npy"))
        # tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))
# 
        # print(layer, np.isclose(torch, tinygrad).sum() / torch.size)

    folder = Path("network")
    for i, layer in enumerate(["net0", "net1", "net2", "fpn0", "fpn1", "fpn2"]):
        torch = np.load(folder.joinpath(layer + "_torch.npy"))
        tinygrad = np.load(folder.joinpath(layer + "_tinygrad.npy"))

        print(layer, np.isclose(torch, tinygrad).sum() / torch.size)
