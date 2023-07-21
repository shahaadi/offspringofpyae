import numpy as np
import random
from matplotlib import pyplot as plt
from graph import connected_components

# node attributes
# label (which cluster it is a part of)
# ID (node's index in the adjacency matrix)
# neighbors (a list of the ID’s of the node’s neighbors)

def propagate_label(node, neighbors, weights, list_of_nodes):
    label_weights = dict()
    for n in neighbors:
        if n.label in label_weights:
            label_weights[n.label] += weights[n.id]
        else:
            label_weights[n.label] = weights[n.id]
    
    node.label = max(zip(label_weights.values(), label_weights.keys()))[1]
    
def whispers(list_of_nodes, num_times):
    x = np.arange(num_times)
    y = []
    for num in num_times:
        node = random.choice(list_of_nodes)
        neighbors = [list_of_nodes[i] for i in node.neighbors]
        propagate_label(node, neighbors, node.weights, list_of_nodes)
        
        """
        print("Iteration #" + num + ":")
        print(connected_components(list_of_nodes))
        print()
        """
        
        y.append(len(connected_components(list_of_nodes)))
    
    # plotting the number of connected_components
    fig, ax = plt.subplots()
    y = np.array(y)
    ax.set_title("Number of Connected Components")
    ax.plot(x, y)
    