import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from math import sqrt, cos
from itertools import product
from PIL import Image
import imageio
import random
import argparse

VECTOR_SIZE = 32
INPUT_SIZE = 2 + 1 + VECTOR_SIZE
DEVICE = "cpu"
COLOR = True
MODE = 0
IMAGE_SIZE = 500
FRAMES = 60
OUTPUT = "out"


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()

        self.layers = []
        first = nn.Linear(INPUT_SIZE, layers[0])
        self.layers.append(first)

        for i in range(1, len(layers)):
            layer = nn.Linear(layers[i - 1], layers[i])
            self.layers.append(layer)

        last = nn.Linear(layers[-1], 1)
        if COLOR:
            last = nn.Linear(layers[-1], 3)
        self.layers.append(last)
        self.layers = nn.ModuleList(self.layers)

        self.funcs = []
        for _ in self.layers:
            n = random.random()
            if n < 0.5:
                self.funcs.append(torch.tanh)
            else:
                self.funcs.append(lambda x: 0.5 * torch.sin(x) + 0.5)

    def forward(self, x):
        for layer, func in zip(self.layers, self.funcs):
            if MODE == 2:
                x = func(layer(x))
            else:
                x = torch.tanh(layer(x))
        if MODE == 2:
            return 0.5 * torch.sin(x) + 0.5
        else:
            return torch.sigmoid(x)


def nn_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=1)


def spec_net(nodes, num_layers):
    layers = []
    for i in range(num_layers):
        layers.append(nodes)

    net = Net(layers)
    net.apply(nn_init)
    net.cuda()
    return net


def config_net(layers):
    net = Net(layers)
    net.apply(nn_init)
    net.cuda()
    return net


def create_pic(net, dim, vector):
    # Normalize the coords
    half = dim / 2
    if MODE == 0 or MODE == 2:
        coords = [
            [
                (x - half) / half,
                (y - half) / half,
                sqrt((x - half) ** 2 + (y - half) ** 2) / sqrt(2 * half ** 2),
            ]
            + vector
            for x, y in product(range(dim), repeat=2)
        ]
    if MODE == 1:
        coords = [
            [
                (x - half) / half,
                (y - half) / half,
                cos(((x - half) ** 2 + (y - half) ** 2)),
            ]
            + vector
            for x, y in product(range(dim), repeat=2)
        ]
    coords = torch.Tensor(coords).cuda()
    out = net(coords).cpu().detach().numpy()
    if COLOR:
        out = out.reshape(dim, dim, 3)
    else:
        out = out.reshape(dim, dim)
    out = (out * 255).astype(np.uint8)
    img = Image.fromarray(out)
    return img


def create_vid(net, dim, frames):
    imgs = []
    # linear interpolation
    p1 = np.random.randn(VECTOR_SIZE)
    p2 = np.random.randn(VECTOR_SIZE)
    dir = (p2 - p1) / frames / 2
    for _ in tqdm(range(frames)):
        imgs.append(create_pic(net, dim, list(p1)))
        p1 += dir
    imageio.mimsave(OUTPUT, imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    format_group = parser.add_mutually_exclusive_group()
    net_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "-m",
        "--mode",
        help="Various modes that will change the look of the output. Enter either 1, 2, or 3.",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--color",
        help="Use black and white or color for the image. 0 for black and white, anything else for color.",
        type=int,
    )
    parser.add_argument(
        "-s",
        "--size",
        help="Size of the output in pixels. Do not set too large otherwise you will run out of memory.",
        type=int,
    )
    parser.add_argument(
        "--gpu",
        help="Use the gpu to speed it up. Will only use the gpu if one is available, even if you set it to true.",
        type=bool,
    )
    parser.add_argument(
        "-f",
        "--frames",
        help="Number of frames to be generated in video flag is passed. Otherwise ignored.",
        type=int,
    )
    parser.add_argument("-o", "--output", help="Path to output.", type=str)
    format_group.add_argument(
        "-p", "--picture", help="Generate a picture as a png.", type=bool
    )
    format_group.add_argument(
        "-v", "--video", help="Generate an animation in gif format.", type=bool
    )
    net_group.add_argument(
        "--spec",
        help="Pass in a string consisting of two space separated positive integers. Look at README for more info.",
        type=str,
    )
    net_group.add_argument(
        "--config",
        help="Pass in a space separated list of integers. Look at README for more info.",
        type=str,
    )

    args = parser.parse_args()
    MODE = args.mode
    if args.color == 0:
        COLOR = 0
    if args.size > 0:
        IMAGE_SIZE = args.size
    else:
        print("Size needs to be greater than 0.")
        exit()
    if args.gpu:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.frames > 0:
        FRAMES = args.frames
    else:
        print("Number of frames needs to be greater than 0.")
        exit()
    OUTPUT = args.output
    if args.spec:
        nodes = int(args.spec.split(" ")[0])
        layers = int(args.spec.split(" ")[1])
        net = spec_net(nodes, layers)
    elif args.config:
        nodes = args.config.split(" ")
        nodes = [int(x) for x in nodes]
        net = config_net(nodes)
    else:
        net = spec_net(16, 4)
    if args.video:
        create_vid(net, IMAGE_SIZE, FRAMES)
    if args.picture:
        vector = np.random.randn(VECTOR_SIZE)
        img = create_pic(net, IMAGE_SIZE, vector)
        imageio.imwrite(OUTPUT, img)

