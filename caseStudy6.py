import numpy as np
import networkx as nx
from collections import Counter

# Exercise 1
def marginal_prob(chars):
    freq_sum = sum(Counter(chars.values()).values())
    m_prob = Counter(chars.values())
    for k,v in m_prob.items():
        m_prob[k] = v/freq_sum
    return dict(m_prob)

def chance_homophily(chars):
    f = list(marginal_prob(chars).values())
    f = np.array(f)
    return (sum(f**2))

favorites_colors = {
        "ankit":"red",
        "xiaoyu":"blue",
        "mary":"blue"
    }
#print(marginal_prob(favorites_colors))
#print(chance_homophily(favorites_colors))

# Exercise 2
import pandas as pd
input_file = "individual_characteristics.csv"
df = pd.read_csv(input_file, low_memory=False, index_col=0)
df1 = df.loc[df["village"]==1]
df2 = df.loc[df["village"]==2]
#print(df1["resp_gend"].head())

# Exercise 3
df11 = df1.set_index("pid")
df22 = df2.set_index("pid")

sex1 = df11.resp_gend.to_dict()
caste1 = df11.caste.to_dict()
religion1 = df11.religion.to_dict()

sex2 = df22.resp_gend.to_dict()
caste2 = df2.caste.to_dict()
religion2 = df22.religion.to_dict()

#print(df22.loc[202802])

#Exercise 4
"""print("Village1 sex c_h is", chance_homophily(sex1))
print("Village1 caste c_h is", chance_homophily(caste1))
print("Village1 religion c_h is", chance_homophily(religion1))

print("Village2 sex c_h is", chance_homophily(sex2))
print("Village2 caste c_h is", chance_homophily(caste2))
print("Village2 religion c_h is", chance_homophily(religion2))"""

# Exercise 5
def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDS,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties = 0
    num_ties = 0
    for n1, n2 in G.edges():
        if IDs[n1] in chars and IDs[n2] in chars:
            if G.has_edge(n1, n2):
                num_ties += 1
                if chars[IDs[n1]] == chars[IDs[n2]]:
                    num_same_ties += 1
    if num_ties > 0:
        return (num_same_ties/num_ties)
    else:
        return 0

# Exercise 6
file1 = "key_vilno_1.csv"
file2 = "key_vilno_2.csv"

pid1 = pd.read_csv(file1)
pid2 = pd.read_csv(file2)

#print(pid1.loc[100])

# Exercise 7
A1 = np.array(pd.read_csv("adj_allVillageRelationships_vilno_1.csv", index_col=0))
A2 = np.array(pd.read_csv("adj_allVillageRelationships_vilno_2.csv", index_col=0))

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

pid1 = pd.read_csv(file1, dtype=int)['0'].to_dict()
pid2 = pd.read_csv(file2, dtype=int)['0'].to_dict()

print("Village1 observed homophily of sex:", homophily(G1, sex1, pid1))
print("Village1 observed homophily of caste:", homophily(G1, caste1, pid1))
print("Village1 observed homophily of religion:", homophily(G1, religion1, pid1))

print("Village1 sex c_h is", chance_homophily(sex1))        # c_h = chance_homophily
print("Village1 caste c_h is", chance_homophily(caste1))
print("Village1 religion c_h is", chance_homophily(religion1))

print()

print("Village2 observed homophily of sex:", homophily(G2, sex2, pid2))
print("Village2 observed homophily of caste:", homophily(G2, caste2, pid2))
print("Village2 observed homophily of religion:", homophily(G2, religion2, pid2))

print("Village2 sex c_h is", chance_homophily(sex2))
print("Village2 caste c_h is", chance_homophily(caste2))
print("Village2 religion c_h is", chance_homophily(religion2))
