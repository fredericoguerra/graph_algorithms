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

def dijkstra_path(graph, origin, end):

    control = {}
    distance = {}
    node_actual = {}
    nodes_not_visited = []
    actual = origin
    node_actual[actual] = 0

    
    for edge in graph.keys():
        nodes_not_visited.append(edge) #inclui os vertices nos não visitados    
        distance[edge] = float('inf') #inicia os vertices como infinito

    distance[actual] = [0,origin] 

    nodes_not_visited.remove(actual)

    while nodes_not_visited:
        for neighbour, cost in graph[actual].items():
             cost_calculated = cost + node_actual[actual]
             if distance[neighbour] == float("inf") or distance[neighbour][0] > cost_calculated:
                 distance[neighbour] = [cost_calculated,actual]
                 control[neighbour] = cost_calculated
                 
        if control == {} : break    
        min_neighbour = min(control.items(), key=lambda x: x[1])
        actual=min_neighbour[0]
        node_actual[actual] = min_neighbour[1]
        nodes_not_visited.remove(actual)
        del control[actual]

    print(f'Minimum distance between {origin} end {end} is: \n{distance[end][0]}')
    print(f"Optimal path is: \n{printPath(distance,origin, end)}")          
    
def printPath(distances,origin, end):
        if  end != origin:
            return "%s -- > %s" % (printPath(distances, origin, distances[end][1]), end)
        else:
            return origin

graph = get_graph(path = './data/dij50.txt')
dijkstra_path(graph,1,50)