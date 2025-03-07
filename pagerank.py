import numpy as np
import networkx as nx

def page_rank(adjancency_matrix, teleportation_probability, max_iterations=100):
    num_nodes = adjancency_matrix.shape[0]
    page_rank_scores = np.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        new_page_rank_scores = adjancency_matrix.dot(page_rank_scores)

        new_page_rank_scores = teleportation_probability + (1 -teleportation_probability) * new_page_rank_scores

        if np.allclose(page_rank_scores, new_page_rank_scores):
            break


    page_rank_scores = new_page_rank_scores

    return page_rank_scores



g = nx.barabasi_albert_graph(2, 1)

page_rank(g, 0.4)