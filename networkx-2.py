import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import itertools

####################################################################################################
num_of_node_graph_1 = 4
attachment_graph_1 = 3

num_of_node_graph_2 = 4
attachment_graph_2 = 3

number_of_weak_links = 3
weight_of_weak_links = 0.1

number_of_strong_links = 1
weight_of_strong_links = 0.01
####################################################################################################

G1 = nx.barabasi_albert_graph(num_of_node_graph_1,attachment_graph_1)
G2 = nx.barabasi_albert_graph(num_of_node_graph_2,attachment_graph_2)
####################################################################################################

def Weight_G1(graph_1):
    x = nx.edges(graph_1)
    W = 10 * np.random.rand(len(x))
    for i in range(len(W)):
        a = float('%.2f' % W[i])
        W[i] = a
    return W
####################################################################################################

def Weight_G2(graph_2):
    x = nx.edges(graph_2)
    W = 10 * np.random.rand(len(x))
    for i in range(len(W)):
        a = float('%.2f' % W[i])
        W[i] = a
    return W
####################################################################################################

def Creat_Graph(graph_1, graph_2):
    H = nx.Graph()
    x1 = nx.nodes(graph_1)
    y1 = nx.edges(graph_1)
    x2 = nx.nodes(graph_2)
    y2 = nx.edges(graph_2)
    x = np.random.rand((len(x1) + len(x2)))
    for i in range(len(x1)):
        x[i] = x1[i]

    for i in range(len(x2)):
        x[i + len(x1)] = x2[i] + len(x1)

    weight_1 = Weight_G1(graph_1)
    weight_2 = Weight_G2(graph_2)

    for i in range(len(y1)):
        t = list(y1[i])
        t.append(weight_1[i])
        a = tuple(t)
        y1[i] = a

    for i in range(len(y2)):
        t = list(y2[i])
        t[0] = t[0] + len(x1)
        t[1] = t[1] + len(x1)
        t.append(weight_2[i])
        a = tuple(t)
        y2[i] = a

    H.add_nodes_from(x)
    H.add_weighted_edges_from(y1)
    H.add_weighted_edges_from(y2)
    return H
####################################################################################################

def Best_weak_link_shortest_path(graph_1, graph_2, num_of_weak_link, weight_of_weak_link):
    x1 = nx.nodes(graph_1)
    x2 = nx.nodes(graph_2)
    for i in range(len(x2)):
        x2[i] = x2[i] + len(x1)

    node_graph_1 = np.copy(x1)
    node_graph_2 = np.copy(x2)
    H = Creat_Graph(graph_1, graph_2)

    x1 = set(x1)
    x2 = set(x2)
    choosen_nodes_of_graph_1 = list(itertools.combinations(x1, num_of_weak_link))
    choosen_nodes_of_graph_2 = list(itertools.combinations(x2, num_of_weak_link))

    weight_of_weaks_connection = 10000
    connection_pairs = np.random.rand(2, 3)

    for i in range(len(choosen_nodes_of_graph_1)):
        weak_link1 = list(choosen_nodes_of_graph_1[i])
        for j in range(len(choosen_nodes_of_graph_2)):
            a = [1, 2, 3]
            b = [1, 2, 3]
            c = [1, 2, 3]

            weak_link2 = list(choosen_nodes_of_graph_2[j])
            a[0] = weak_link1[0]
            b[0] = weak_link1[1]
            c[0] = weak_link1[2]

            a[1] = weak_link2[0]
            b[1] = weak_link2[1]
            c[1] = weak_link2[2]
            a[2] = b[2] = c[2] = weight_of_weak_link
            a = tuple(a)
            b = tuple(b)
            c = tuple(c)
            d = [a, b, c]

            H.add_weighted_edges_from(d)
            weight_of_weakth_connection = 0

            for n in range(len(node_graph_1)):
                min_one_node_W = 0
                for m in range(len(node_graph_2)):
                    W = nx.nx.shortest_path_length(H,node_graph_1[n], node_graph_2[m], 'weight')
                    min_one_node_W = min_one_node_W + W
                weight_of_weakth_connection = weight_of_weakth_connection + min_one_node_W

            if weight_of_weakth_connection< weight_of_weaks_connection:
                weight_of_weaks_connection = weight_of_weakth_connection
                connection_pairs[0] = list(choosen_nodes_of_graph_1[i])
                connection_pairs[1] = list(choosen_nodes_of_graph_2[j])

            for l in d:
                H.remove_edge(*l[:2])
    print(weight_of_weaks_connection)
    return connection_pairs

    # print(connection_pairs)
    # print(weight_of_weaks_connection)
    # nx.draw(H, with_labels=True)
    # plt.show()
####################################################################################################

def Best_strong_link_shortest_path(graph_1, graph_2, num_of_strong_link, weight_of_strong_link):
    x1 = nx.nodes(graph_1)
    x2 = nx.nodes(graph_2)
    for i in range(len(x2)):
        x2[i] = x2[i] + len(x1)

    node_graph_1 = np.copy(x1)
    node_graph_2 = np.copy(x2)
    H = Creat_Graph(graph_1, graph_2)

    x1 = set(x1)
    x2 = set(x2)
    choosen_nodes_of_graph_1 = list(itertools.combinations(x1, num_of_strong_link))
    choosen_nodes_of_graph_2 = list(itertools.combinations(x2, num_of_strong_link))

    weight_of_strongs_connection = 10000
    connection_pairs = np.random.rand(2, 3)

    for i in range(len(choosen_nodes_of_graph_1)):
        strong_link1 = list(choosen_nodes_of_graph_1[i])
        for j in range(len(choosen_nodes_of_graph_2)):
            a = [1, 2, 3]

            strong_link2 = list(choosen_nodes_of_graph_2[j])
            a[0] = strong_link1[0]

            a[1] = strong_link2[0]

            a[2] = weight_of_strong_link
            a = tuple(a)
            d = [a]

            H.add_weighted_edges_from(d)
            weight_of_strongth_connection = 0

            for n in range(len(node_graph_1)):
                min_one_node_W = 0
                for m in range(len(node_graph_2)):
                    W = nx.nx.shortest_path_length(H,node_graph_1[n], node_graph_2[m], 'weight')
                    min_one_node_W = min_one_node_W + W
                weight_of_strongth_connection = weight_of_strongth_connection + min_one_node_W

            if weight_of_strongth_connection< weight_of_strongs_connection:
                weight_of_strongs_connection = weight_of_strongth_connection
                connection_pairs[0] = list(choosen_nodes_of_graph_1[i])
                connection_pairs[1] = list(choosen_nodes_of_graph_2[j])

            for l in d:
                H.remove_edge(*l[:2])
    print(weight_of_strongs_connection)
    return connection_pairs

    # print(connection_pairs)
    # print(weight_of_strongs_connection)
    # nx.draw(H, with_labels=True)
    # plt.show()
####################################################################################################

# NODES = Best_strong_link_shortest_path(G1, G2, number_of_strong_links, weight_of_strong_links)
NODES = Best_weak_link_shortest_path(G1, G2, number_of_weak_links, weight_of_weak_links)
print(NODES)
G = Creat_Graph(G1, G2)
b = []

for i in range(3):
    a = np.random.rand(3)
    a[0] = NODES[0][i]
    a[1] = NODES[1][i]
    a[2] = weight_of_weak_links
    a = tuple(a)
    b.append(a)

G.add_weighted_edges_from(b)


pos=nx.spring_layout(G)
nx.draw(G, pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()



