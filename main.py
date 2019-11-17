# Implementation for Project 5 - option 1: Virus Propagation on Static Networks

# Project Team Members:
# 1. Wen-Han Hu (whu24)
# 2. Yang-Kai Chou (ychou3)
# 3. Yuan Xu (yxu48)

import networkx as nx
import numpy as np


def main():
    # Read the data from file
    g = nx.Graph()
    file = './data/static.network'
    with open(file) as f:
        next(f)
        for line in f:
            line = line.split()
            g.add_edge(int(line[0]), int(line[1]))
    print(len(g.edges))


if __name__ == "__main__":
    main()
