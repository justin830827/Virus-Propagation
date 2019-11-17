# Implementation for Project 5 - option 1: Virus Propagation on Static Networks

# Project Team Members:
# 1. Wen-Han Hu (whu24)
# 2. Yang-Kai Chou (ychou3)
# 3. Yuan Xu (yxu48)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
from operator import itemgetter

random.seed(123)

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


def compute_eigen(adj):
    eigenvalue, eigenvector = np.linalg.eig(adj)
    eig_set = [(eigenvalue[i], eigenvector[i]) for i in range(len(eigenvalue))]
    eig_set = sorted(eig_set, key=lambda x: x[0], reverse=1)
    return eig_set


def policyA(adj, g, k=200):
    immun = set()
    total_node = nx.number_of_nodes(g)
    while len(immun) < k:
        r = random.randint(0, total_node-1)
        if r not in immun:
            immun.add(r)

    for node in immun:
        for i in range(len(adj[node])):
            adj[node][i] = 0
            adj[i][node] = 0
    return adj


def policyB(adj, g, k=200):
    immun = set()
    degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
    for i in range(k):
        immun.add(degree[i][0])

    for node in immun:
        for i in range(len(adj[node])):
            adj[node][i] = 0
            adj[i][node] = 0
    return adj


def policyC(adj, g, k=200):
    immun = set()
    total_node = nx.number_of_nodes(g)
    while len(immun) < k:
        degree = sorted(list(g.degree()), key=lambda x: x[1], reverse=1)
        immun.add(degree[0][0])
        g.remove_node(degree[0][0])

    adjC = [[0 for _ in range(total_node)] for _ in range(total_node)]
    for i in nx.edges(g):
        adjC[i[0]][i[1]] = 1
        adjC[i[1]][i[0]] = 1
    return adjC


def policyD(adj, g, k=200):
    total_node = nx.number_of_nodes(g)
    val, vec = np.linalg.eig(adj)
    eig_set = [(val[i], vec[i]) for i in range(len(val))]
    eig_set = sorted(eig_set, key=lambda x: x[0], reverse=1)
    largest_vec = eig_set[0][1]
    largest_vec = np.absolute(largest_vec)
    target = [u[0]
              for u in sorted(enumerate(largest_vec), reverse=True, key=itemgetter(1))[:k]]
    for i in target:
        g.remove_node(i)
    adjD = [[0 for _ in range(total_node)] for _ in range(total_node)]
    for i in nx.edges(g):
        adjD[i[0]][i[1]] = 1
        adjD[i[1]][i[0]] = 1
    return adjD


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
    print("Computing Strength................")
    # create adjacency matrix
    adj = [[0 for _ in range(total_node)] for _ in range(total_node)]
    for i in nx.edges(g):
        adj[i[0]][i[1]] = 1
        adj[i[1]][i[0]] = 1

    # # a. Evaluate infection spread across the network
    # eigen_set = compute_eigen(adj)
    # largest_eigenvalue = eigen_set[0][0].real
    # s1 = largest_eigenvalue * Cvpm1
    # s2 = largest_eigenvalue * Cvpm2
    # print("Effective strengh with beta = {}, delta = {} is: {}".format(
    #     beta1, delta1, s1))
    # print("Effective strengh with beta = {}, delta = {} is: {}\n".format(
    #     beta2, delta2, s2))

    # # b. Fix delta and evaluate bata affecting the effective strength
    # # delta1 retuls
    # fixed_delta1_min_beta, fixed_delta1_strength = static_delta(
    #     largest_eigenvalue, delta1)
    # print("Minimum transmission probability (β): {} with fixed delta = {}".format(
    #     fixed_delta1_min_beta, delta1))
    # plot_static(fixed_delta1_strength, 'Beta',
    #             'Strength vs varying beta with delta = {}'.format(delta1))
    # # delta2 retuls
    # fixed_delta2_min_beta, fixed_delta2_strength = static_delta(
    #     largest_eigenvalue, delta2)
    # print("Minimum transmission probability (β): {} with fixed delta = {}\n".format(
    #     fixed_delta2_min_beta, delta2))
    # plot_static(fixed_delta2_strength, 'Beta',
    #             'Strength vs varying beta with delta = {}'.format(delta2))

    # # c. Fix beta and evaluate delta affecting the effective strength
    # # beta1 results
    # fixed_beta1_max_delta, fixed_beta1_strength = static_beta(
    #     largest_eigenvalue, beta1)
    # print("Maximum healing probability (δ): {} with fixed beta = {}".format(
    #     fixed_beta1_max_delta, beta1))
    # plot_static(fixed_beta1_strength, 'Beta',
    #             'Strength vs varying delta with beta = {}'.format(beta1))
    # # beta2 results
    # fixed_beta2_max_delta, fixed_beta2_strength = static_beta(
    #     largest_eigenvalue, beta2)
    # print("Maximum healing probability (δ): {} with fixed beta = {}\n".format(
    #     fixed_beta2_max_delta, beta2))
    # plot_static(fixed_beta2_strength, 'Beta',
    #             'Strength vs varying delta with beta = {}'.format(beta2))

    # # Part 2: Simulates the propagation of virus with the SIS VPM
    # print('Simulating the virus propagation with β = {}, δ = {}.........'.format(
    #     beta1, delta1))
    # # beta1 and delta1 results
    # first_series = []
    # for i in range(10):
    #     res = simulation(g, beta1, delta1, adj)
    #     first_series.append(res)
    # plot_simulation(first_series, total_node, 'Simulation with β = {} and δ = {}'.format(
    #     beta1, delta1))

    # print('Simulating the virus propagation with β = {}, δ = {}.........\n'.format(
    #     beta2, delta2))
    # # beta2 and delta2 results
    # second_series = []
    # for i in range(10):
    #     res = simulation(g, beta2, delta2, adj)
    #     second_series.append(res)
    # plot_simulation(second_series, total_node, 'Simulation with β = {} and δ = {}'.format(
    #     beta2, delta2))

    # Part 3:  Implements immunization policies
    # d. calculate the effective strength (s) of the virus on the immunized contact network
    print("\n\nImmunization wiht Policy A.....................")
    adj_matrixA = policyA(adj.copy(), g.copy())
    eigen_set_A = compute_eigen(adj_matrixA)
    largest_eigenvalue_A = eigen_set_A[0][0].real
    sA_1 = largest_eigenvalue_A * Cvpm1
    sA_2 = largest_eigenvalue_A * Cvpm2
    print("Effective strengh of Policy A with beta = {}, delta = {} is: {}".format(
        beta1, delta1, sA_1))
    print("Effective strengh of Policy A with beta = {}, delta = {} is: {}\n".format(
        beta2, delta2, sA_2))

    print("\n\nImmunization wiht Policy B.....................")
    adj_matrixB = policyB(adj.copy(), g.copy())
    eigen_set_B = compute_eigen(adj_matrixB)
    largest_eigenvalue_B = eigen_set_B[0][0].real
    sB_1 = largest_eigenvalue_B * Cvpm1
    sB_2 = largest_eigenvalue_B * Cvpm2
    print("Effective strengh of Policy B with beta = {}, delta = {} is: {}".format(
        beta1, delta1, sB_1))
    print("Effective strengh of Policy B with beta = {}, delta = {} is: {}\n".format(
        beta2, delta2, sB_2))

    print("\n\nImmunization wiht Policy C.....................")
    adj_matrixC = policyC(adj.copy(), g.copy())
    eigen_set_C = compute_eigen(adj_matrixC)
    largest_eigenvalue_C = eigen_set_C[0][0].real
    sC_1 = largest_eigenvalue_C * Cvpm1
    sC_2 = largest_eigenvalue_C * Cvpm2
    print("Effective strengh of Policy C with beta = {}, delta = {} is: {}".format(
        beta1, delta1, sC_1))
    print("Effective strengh of Policy C with beta = {}, delta = {} is: {}\n".format(
        beta2, delta2, sC_2))

    print("\n\nImmunization wiht Policy D.....................")
    adj_matrixD = policyD(adj.copy(), g.copy())
    eigen_set_D = compute_eigen(adj_matrixD)
    largest_eigenvalue_D = eigen_set_D[0][0].real
    sD_1 = largest_eigenvalue_D * Cvpm1
    sD_2 = largest_eigenvalue_D * Cvpm2
    print("Effective strengh of Policy D with beta = {}, delta = {} is: {}".format(
        beta1, delta1, sD_1))
    print("Effective strengh of Policy D with beta = {}, delta = {} is: {}\n".format(
        beta2, delta2, sD_2))


if __name__ == "__main__":
    main()
