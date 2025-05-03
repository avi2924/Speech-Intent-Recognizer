import matplotlib.pyplot as plt
import networkx as nx

# Create a bipartite graph
G = nx.Graph()

# Define nodes
bus_stops = ['1', '2', '3', '4', '7', '8', '9', '10', '11']
buses = ['Bus1', 'Bus2', 'Bus3', 'Bus4']

# Add nodes with bipartite attribute
G.add_nodes_from(bus_stops, bipartite=0)  # bus stops
G.add_nodes_from(buses, bipartite=1)      # buses

# Add edges between bus stops and buses
edges = [
    ('1', 'Bus1'), ('3', 'Bus1'), ('7', 'Bus1'),
    ('4', 'Bus2'), ('7', 'Bus2'), ('9', 'Bus2'), ('10', 'Bus2'),
    ('2', 'Bus3'), ('1', 'Bus3'), ('10', 'Bus3'),
    ('8', 'Bus4'), ('4', 'Bus4'), ('11', 'Bus4'), ('10', 'Bus4')
]
G.add_edges_from(edges)

# Position nodes using bipartite layout
pos = {}
pos.update((node, (0, i)) for i, node in enumerate(bus_stops))  # bus stops on the left
pos.update((node, (1, i)) for i, node in enumerate(buses))      # buses on the right

plt.figure(figsize=(12, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=bus_stops, node_color='lightgreen', node_size=1000, label='Bus Stops')
nx.draw_networkx_nodes(G, pos, nodelist=buses, node_color='skyblue', node_size=1000, label='Buses')

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=edges)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

# Legend and title
plt.legend(scatterpoints=1, markerscale=0.01)
plt.title('Modified Graph (Bipartite Graph: Bus Stops ↔ Buses)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()
