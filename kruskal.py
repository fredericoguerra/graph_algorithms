from quick_sort import partition, quick_sort
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
    final_row = [0]*int(matrix_len)
    sup_matrix.append(final_row)


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

def partition(A: list,start,end):
    pivot_index = start
    pivot = cost(graph,A[start])
    while start < end:
        while start < len(A) and cost(graph,A[start]) <= cost(graph,A[pivot_index]):
            start += 1
        while cost(graph,A[end]) > pivot:
            end -= 1
        if (start < end):
            A[start], A[end] = A[end], A[start]
    
    A[end], A[pivot_index] = A[pivot_index], A[end]
    return end

def cost(graph, edge):
    return graph[edge[0]][edge[1]]

def quick_sort(A: list, start: int, end: int):
    if (start<end):
        p = partition(A, start, end)
        quick_sort(A, start, p - 1)
        quick_sort(A, p + 1, end)
    return A

def order_edges(graph):
    ordered_edges = []
    for u,neighbour in graph.items():
        for v, cost in neighbour.items():
            ordered_edges.append(tuple([u, v]))
    ordered_edges = quick_sort(ordered_edges,0,len(ordered_edges)-1)
    return ordered_edges

sets = {}

def union(x, y):
    xRepresentative = find(x)
    yRepresentative = find(y)
    sets[yRepresentative] = sets[yRepresentative].union(sets[xRepresentative])
    del sets[xRepresentative]

def makeSet(x):
    sets[x] = set([x])

def find(x):
    for representative,subset in sets.items():
        if x in subset:
            return representative
    return None

def kruskal(graph):
    ordered_edges = order_edges(graph)
    tree_min = []
    for v in graph.keys():
        makeSet(v)
    for edge in ordered_edges:
        if find(edge[0]) != find(edge[1]):
            tree_min.append(edge)
            union(edge[0], edge[1])

    return tree_min

graph = get_graph(path = './data/dij50.txt')
tree_min = kruskal(graph)

print(tree_min)

cost = 0
for i in tree_min:
    cost += graph[i[0]][i[1]]

print('total_cost:',cost)