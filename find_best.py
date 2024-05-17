import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.pyplot import MultipleLocator
import argparse


def parse_file(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []
    pattern = "Average elasped time: \((.*?)\) second, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        r = re.match(pattern, line)
        if r:
            gflops = float(r.group(2))
            data.append(gflops)
    return data


def plot(data, save_dir):
    max_len = min([len(v) for v in data.values()])
    x = [(i + 1) * 256 for i in range(max_len)]
    fig = plt.figure(figsize=(12, 10))

    for name, y in data.items():
        plt.plot(x, y[:max_len], linewidth=2, label=name)
        plt.scatter(x, y[:max_len], marker="s", s=60, edgecolors='k', linewidth=2)

    plt.legend()

    plt.tick_params(labelsize=10, rotation=45)
    plt.xlabel("Matrix size (M=N=K)", fontsize=12, fontweight='bold')
    plt.ylabel("Performance (GFLOPS)", fontsize=12, fontweight='bold')

    plt.title(f"Impact of different BM BN BK TM TN have on GEMM performance", fontsize=16, fontweight='bold')

    x_major_locator = MultipleLocator(256)
    plt.gca().xaxis.set_major_locator(x_major_locator)

    plt.savefig(f"{save_dir}/kernel_autotune.png")


def main(args):
    root = os.path.dirname(os.path.abspath(__file__))
    paths = Path(root).glob("test/autotune/autotune_*.txt")
    data = {}
    for path in paths:
        name = re.search(r"autotune_(\d+_\d+_\d+_\d+_\d+).txt", str(path)).group(1)
        datum = parse_file(path)
        if datum and len(datum) >= 16:
            data[name] = datum
    plot(data, args.save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='plot kernel performance')
    parser.add_argument('--save_dir', default='images')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python plot.py 0 1
