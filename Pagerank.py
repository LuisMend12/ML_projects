import networkx as nx
G = nx.barabasi_albert_graph(2, 1)
pr=nx.pagerank(G, 0.4)
pr