import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli

input_file1 = "adj_allVillageRelationships_vilno_1.csv"
input_file2 = "adj_allVillageRelationships_vilno_2.csv"

#G = nx.karate_club_graph()
#nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
#plt.savefig("node.pdf")
#print(G.degree())
#plt.show()

# Custom ER model
def er_graph(N, p):
    """Generate an ER graph"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G
#N=20
#p=0.2
#nx.draw(er_graph(50, 0.08), node_size=40, node_color="gray")
#plt.show()

#Plotting degreed distribution
def plot_degree_distribution(G):
    plt.hist(list(dict(G.degree()).values()), histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree Distribution")

"""G1 = er_graph(500, 0.08)
plot_degree_distribution(G1)
G2 = er_graph(500, 0.08)
plot_degree_distribution(G2)
G3 = er_graph(500, 0.08)
plot_degree_distribution(G3)
plt.savefig("hist_3.pdf")
plt.show()"""

A1 = np.loadtxt(input_file1, delimiter=',')
A2 = np.loadtxt(input_file2, delimiter=',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    print("Average degree: %0.2f" % np.mean(list(dict(G.degree()).values())))

basic_net_stats(G1)
basic_net_stats(G2)
"""plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village_hist.pdf")
plt.show()"""

# Finding largest connected component
gen_G1 = (G1.subgraph(n) for n in nx.connected_components(G1))
gen_G2 = (G2.subgraph(n) for n in nx.connected_components(G2))
G1_LCC = max(gen_G1, key=len)
G2_LCC = max(gen_G2, key=len)
#print(len(G1_LCC))
# Percentage of nodes in LCC
#print(G1_LCC.number_of_nodes()/G1.number_of_nodes())
#print(G2_LCC.number_of_nodes()/G2.number_of_nodes())

"""plt.figure()
nx.draw(G1_LCC, node_color="red", edge_color="gray", node_size=20)
plt.savefig("village1.pdf")
plt.figure()
nx.draw(G2_LCC, node_color="green", edge_color="gray", node_size=20)
plt.savefig("village2.pdf")
plt.show()"""
