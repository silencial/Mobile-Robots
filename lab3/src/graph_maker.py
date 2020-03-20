import networkx as nx
import pickle
import os
import numpy as np
import multiprocessing
from multiprocessing import Pool

assert (nx.__version__ == '2.2' or nx.__version__ == '2.1')
PARALLEL = True


def load_graph(filename):
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print('Loaded graph from {}'.format(f))
    return data['G']


def connect_edges(arg):
    start, end, valid_vertices, env, connection_radius = arg
    edges = []
    for i in range(start, end):
        distances = env.compute_distances(valid_vertices[i], valid_vertices[i+1:])
        for j, dis in enumerate(distances):
            if dis <= connection_radius and env.edge_validity_checker(valid_vertices[i], valid_vertices[i+j+1])[0]:
                edges.append((i, i+j+1, dis))
    return edges


def make_graph(env, sampler, connection_radius, num_vertices, directed=True, lazy=False, saveto='graph.pkl'):
    """
    Returns a graph on the passed environment.
    All vertices in the graph must be collision-free.

    Graph should have node attribute "config" which keeps a configuration in tuple.
    E.g., for adding vertex "0" with configuration np.array([0, 1]),
    G.add_node(0, config=tuple(config))

    To add edges to the graph, call
    G.add_weighted_edges_from(edges)
    where edges is a list of tuples (node_i, node_j, weight),
    where weight is the distance between the two nodes.

    @param env: Map Environment for graph to be made on
    @param sampler: Sampler to sample configurations in the environment
    @param connection_radius: Maximum distance to connect vertices
    @param num_vertices: Minimum number of vertices in the graph.
    @param lazy: If true, edges are made without checking collision.
    @param saveto: File to save graph and the configurations
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # TODO
    # 1. Sample vertices
    # 2. Connect them with edges
    vertices = sampler.sample(num_vertices)
    valid_vertices = []
    ind = 0
    for config in vertices:
        if env.state_validity_checker(config):
            G.add_node(ind, config=tuple(config))
            valid_vertices.append(config)
            ind += 1
    valid_vertices = np.array(valid_vertices)

    if lazy:
        edges = []
        for i, config1 in enumerate(valid_vertices):
            distances = env.compute_distances(config1, valid_vertices[i+1:])
            for j, dis in enumerate(distances):
                if dis <= connection_radius:
                    edges.append((i, i+j+1, dis))
        G.add_weighted_edges_from(edges)
    else:
        if PARALLEL:
            threads = 8
            args = []
            total = len(valid_vertices)
            prev = 0
            b = 2*total + 1
            seg = float(total) * (total+1) / threads
            for i in range(threads):
                start = prev
                if i == threads - 1:
                    end = total
                else:
                    c = (2*total - start + 1) * start + seg
                    end = int((b - (b**2 - 4*c)**0.5) / 2)
                # print(start, end, total)
                args.append((start, end, valid_vertices, env, connection_radius))
                prev = end
            cores = multiprocessing.cpu_count()
            pool = Pool(processes=cores)
            edges = pool.map(connect_edges, args)
            for i in edges:
                G.add_weighted_edges_from(i)
        else:
            edges = []
            for i, config1 in enumerate(valid_vertices):
                distances = env.compute_distances(config1, valid_vertices[i+1:])
                for j, dis in enumerate(distances):
                    if dis <= connection_radius and env.edge_validity_checker(config1, valid_vertices[i+j+1])[0]:
                        edges.append((i, i+j+1, dis))
            G.add_weighted_edges_from(edges)

    # Save the graph to reuse.
    if saveto is not None:
        data = dict(G=G)
        pickle.dump(data, open(saveto, 'wb'))
        print('Saved the graph to {}'.format(saveto))
    return G


def add_node(G, config, env, connection_radius, start_from_config):
    """
    This function should add a node to an existing graph G.
    @param G graph, constructed using make_graph
    @param config Configuration to add to the graph
    @param env Environment on which the graph is constructed
    @param connection_radius Maximum distance to connect vertices
    @param start_from_config True if config is the starting configuration
    """
    # new index of the configuration
    index = G.number_of_nodes()
    G.add_node(index, config=tuple(config))
    G_configs = nx.get_node_attributes(G, 'config')
    G_configs = [G_configs[node] for node in G_configs]

    # TODO
    # Add edges from the newly added node
    edges = []
    distances = env.compute_distances(config, G_configs)
    for i, dis in enumerate(distances):
        if dis <= connection_radius and env.edge_validity_checker(config, G_configs[i])[0]:
            edges.append((i, index, dis))
    G.add_weighted_edges_from(edges)

    # Check for connectivity.
    num_connected_components = len(list(nx.connected_components(G)))
    if num_connected_components != 1:
        print("warning, Graph has {} components, not connected".format(num_connected_components))

    return G, index
