# Implementation for Project 5 - option 1: Virus Propagation on Static Networks

# Project Team Members:
# 1. Wen-Han Hu (whu24)
# 2. Yang-Kai Chou (ychou3)
# 3. Yuan Xu (yxu48)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# configuration
beta1 = 0.2
beta2 = 0.01
delta1 = 0.7
delta2 = 0.6
Cvpm1 = beta1 / delta1
Cvpm2 = beta2 / delta2


def static_delta(eigenval, delta):
    static_delta = []
    min_beta = 0
    flag = 0
    print("Computing minimum transmission probability (β) with delta = {}".format(delta))
    for i in range(1, 1001):
        tmp_beta = float(i) / 1000
        tmp_Cvpm = tmp_beta / delta
        strength = eigenval * tmp_Cvpm
        # get the minimum beta to cause an epidemic
        if strength > 1 and flag == 0:
            min_beta = tmp_beta
            flag = 1
        static_delta.append(strength)
    return min_beta, static_delta


def static_beta(eigenval, beta):
    static_beta = []
    max_delta = 1
    flag = 0
    print("Computing maximum healing probability (δ) with beta = {}".format(beta))
    for i in range(1, 1001):
        tmp_delta = float(i) / 1000
        tmp_Cvpm = beta / tmp_delta
        strength = eigenval * tmp_Cvpm
        if strength < 1 and flag == 0:
            max_delta = tmp_delta
            flag = 1
        static_beta.append(strength)
    return max_delta, static_beta


def plot_static(strength, param_name, title):
    param_range = np.arange(0.001, 1.001, 0.001)
    plt.plot(param_range, strength)
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Effective Strength')
    plt.axhline(y=1, linewidth=2, color='r')
    plt.savefig('results/{}.png'.format(title), bbox_inches='tight')
    plt.close()


def simulation(g, beta, delta, adj, t=100):
    total_node = nx.number_of_nodes(g)
    c = int(total_node/10)
    infect = [False for _ in range(total_node)]
    infect_ind = []
    while len(infect_ind) < c:
        r = random.randint(0, total_node-1)
        if infect[r] == False:
            infect[r] = True
            infect_ind.append(r)

    infect_num = []
    infect_num.append(len(infect_ind))

    for i in range(t):
        cur_infect = set()
        cur_cure = []
        for inf in infect_ind:
            for j in range(len(adj[inf])):
                if adj[inf][j] == 1 and float(random.randint(1, 10))/10 < beta:
                    cur_infect.add(j)
            if float(random.randint(1, 10))/10 < delta:
                cur_cure.append(inf)
            else:
                cur_infect.add(inf)

        for node in cur_cure:
            infect[node] = False
        for node in cur_infect:
            infect[node] = True
        infect_num.append(len(cur_infect))
        infect_ind = cur_infect
    return infect_num


def plot_simulation(series, total_node, title):
    series = np.array(series) / total_node
    series = np.mean(series, axis=0)
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Times')
    plt.ylabel('Avg. fraction of infected nodes')
    plt.savefig('results/{}.png'.format(title), bbox_inches='tight')
    plt.close()


def compute_strength(g):
    print("Computing Strength................")
    total_node = nx.number_of_nodes(g)
    # create adj_matrix
    adj = [[0 for _ in range(total_node)] for _ in range(total_node)]
    for i in nx.edges(g):
        adj[i[0]][i[1]] = 1
        adj[i[1]][i[0]] = 1
    eigenvalue, eigenvector = np.linalg.eig(adj)
    eig_set = [(eigenvalue[i], eigenvector[i]) for i in range(len(eigenvalue))]
    eig_set = sorted(eig_set, key=lambda x: x[0], reverse=1)
    return eig_set, adj


def main():
    # Read the data from file
    g = nx.Graph()
    file = './data/static.network'
    print("Read static network...............")
    with open(file) as f:
        next(f)
        for line in f:
            line = line.split()
            g.add_edge(int(line[0]), int(line[1]))

    # Part 1: calculate the effective strength (s)
    total_node = nx.number_of_nodes(g)
    eigen_set, adj_matrix = compute_strength(g)

    # a. Evaluate infection spread across the network
    largest_eigenvalue = eigen_set[0][0].real
    s1 = largest_eigenvalue * Cvpm1
    s2 = largest_eigenvalue * Cvpm2
    print("Effective strengh with beta = {}, delta = {} is: {}".format(
        beta1, delta1, s1))
    print("Effective strengh with beta = {}, delta = {} is: {}\n".format(
        beta2, delta2, s2))

    # b. Fix delta and evaluate bata affecting the effective strength
    # delta1 retuls
    fixed_delta1_min_beta, fixed_delta1_strength = static_delta(
        largest_eigenvalue, delta1)
    print("Minimum transmission probability (β): {} with fixed delta = {}".format(
        fixed_delta1_min_beta, delta1))
    plot_static(fixed_delta1_strength, 'Beta',
                'Strength vs varying beta with delta = {}'.format(delta1))
    # delta2 retuls
    fixed_delta2_min_beta, fixed_delta2_strength = static_delta(
        largest_eigenvalue, delta2)
    print("Minimum transmission probability (β): {} with fixed delta = {}\n".format(
        fixed_delta2_min_beta, delta2))
    plot_static(fixed_delta2_strength, 'Beta',
                'Strength vs varying beta with delta = {}'.format(delta2))

    # c. Fix beta and evaluate delta affecting the effective strength
    # beta1 results
    fixed_beta1_max_delta, fixed_beta1_strength = static_beta(
        largest_eigenvalue, beta1)
    print("Maximum healing probability (δ): {} with fixed beta = {}".format(
        fixed_beta1_max_delta, beta1))
    plot_static(fixed_beta1_strength, 'Beta',
                'Strength vs varying delta with beta = {}'.format(beta1))
    # beta2 results
    fixed_beta2_max_delta, fixed_beta2_strength = static_beta(
        largest_eigenvalue, beta2)
    print("Maximum healing probability (δ): {} with fixed beta = {}\n".format(
        fixed_beta2_max_delta, beta2))
    plot_static(fixed_beta2_strength, 'Beta',
                'Strength vs varying delta with beta = {}'.format(beta2))

    # Part 2: Simulates the propagation of virus with the SIS VPM
    print('Simulating the virus propagation with β = {}, δ = {}.........'.format(
        beta1, delta1))
    # beta1 and delta1 results
    first_series = []
    for i in range(10):
        res = simulation(g, beta1, delta1, adj_matrix)
        first_series.append(res)
    plot_simulation(first_series, total_node, 'Simulation with β = {} and δ = {}'.format(
        beta1, delta1))

    print('Simulating the virus propagation with β = {}, δ = {}.........\n'.format(
        beta2, delta2))
    # beta2 and delta2 results
    second_series = []
    for i in range(10):
        res = simulation(g, beta2, delta2, adj_matrix)
        second_series.append(res)
    plot_simulation(second_series, total_node, 'Simulation with β = {} and δ = {}'.format(
        beta2, delta2))


if __name__ == "__main__":
    main()
