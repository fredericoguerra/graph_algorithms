import random
import numpy as np

def get_graph(path: str):
    with open(path) as f:
        lines = f.readlines()

    matrix_len = lines[0]
    superior_ele = lines[2:]
    sup_matrix = []

    for v in superior_ele:
        sup_matrix.append(list(np.fromstring(v,dtype=int,sep=' ')))

    for i in range(0,int(matrix_len)-1):
        add_eles = [0]*(i+1)
        sup_matrix[i] = add_eles + sup_matrix[i]
    sup_matrix.append([0]*int(matrix_len))

    adj_matrix = np.array(sup_matrix)
    adj_matrix += adj_matrix.T
    graph = {}

    for row in range(0,adj_matrix.shape[0]):
        for col in range(0,adj_matrix.shape[1]):
            if col != row:
                if row+1 in graph.keys():
                    graph[row+1][col+1] =  adj_matrix[row][col]
                else:
                    graph[row+1] = {col+1: adj_matrix[row][col]}

    return graph

def prim_algorithm(graph):
    root = random.randint(1,len(graph))
    selected_nodes = [root]
    iterations = []

    while (len(selected_nodes) != len(graph)):
        min_value = 100000000
        child_node = None
        parent_node = None
        res = [i for i in selected_nodes if i]
        for sel_node in res:
            for node in graph[sel_node]:
                if node not in res and graph[sel_node][node]<min_value:
                    child_node = node
                    parent_node = sel_node
                    min_value = graph[sel_node][node]
        selected_nodes.append(child_node)
        iterations.append((parent_node,child_node))

    return iterations, selected_nodes

graph = get_graph(path='./data/dij50.txt')

iterations, T = prim_algorithm(graph)
print(iterations,"\n",T)
cost = 0
for i in iterations:
    cost += graph[i[0]][i[1]]

print('total_cost:',cost)