from matplotlib import pyplot as plt
import os
from glob import glob
from graph import Node, makeGraph
from whisper_function import whispers
import numpy as np
import random

# get file_paths and display graph with all pictures
file_paths = glob(os.path.join("./week2 - y/people_pictures", "*.jpg"))

images = []

c = 3
while c < len(file_paths):
    del file_paths[c]
    c += 3

random.shuffle(file_paths)

for i in range(0, len(file_paths)):
    images.append(plt.imread(file_paths[i]))

rows = 3
cols = 10
pic_num = 0

for y in range(0, cols):
    for i in range(0, rows):
        plt.subplot(rows, cols, i * cols + y + 1)
        plt.imshow(images[pic_num])
        plt.axis('off')
        """
        if pic_num % rows == 0:
            result = str(file_paths[pic_num])
            result = result[28:len(result) - 5]
            plt.title(result)
        """
        
        pic_num += 1
        
plt.show()


# run the whisper function
list_of_nodes = makeGraph(file_paths, cos_dist_threshold=0.9, face_prob_threshold=0.9)
x_values, y_values = whispers(list_of_nodes, 150)
# plotting the number of connected_components
plt.plot(x_values, y_values)
plt.title("Number of Connected Components")
plt.show()

"""
fig, ax = plt.subplots()
y_values = np.array(y_values)
ax.set_title("Number of Connected Components")
ax.plot(x_values, y_values)
"""


# plot the result of the whisper function by category
labels_list = []
ordered_nodes_list = []
for n in list_of_nodes:
    if n.label in labels_list:
        ordered_nodes_list[labels_list.index(n.label)].append(n)
    else:
        ordered_nodes_list.append([n])
        labels_list.append(n.label)

max_len = 0
for no in ordered_nodes_list:
    if len(no) > max_len:
        max_len = len(no)
        
rows = max_len
cols = len(ordered_nodes_list)
for y in range(0, cols):
    for i in range(0, len(ordered_nodes_list[y])):
        plt.subplot(rows, cols, i * cols + y + 1)
        plt.tight_layout()
        image = plt.imread(ordered_nodes_list[y][i].file_path)
        plt.imshow(image)
        plt.axis('off')
        if i == 0:
            result = str(ordered_nodes_list[y][i].file_path)
            result = result[28:len(result) - 5]
            """
            result2 = str(ordered_nodes_list[y][i].label)
            result = result + "(" + result2 + ")"
            print(result)
            """
            plt.title(result)
plt.show()

"""
# testing final graph
# run the whisper function
one = ('./week2 - y/people_pictures\\beyonce1.jpg', 0)
two = ('./week2 - y/people_pictures\\jackiechan4.jpg', 1)
three = ('./week2 - y/people_pictures\\rihanna3.jpg', 2)
four = ('./week2 - y/people_pictures\\beyonce2.jpg', 0)
five = ('./week2 - y/people_pictures\\willsmith1.jpg', 3)
six = ('./week2 - y/people_pictures\\willsmith2.jpg', 3)
seven = ('./week2 - y/people_pictures\\rihanna2.jpg', 2)
eight = ('./week2 - y/people_pictures\\jackiechan4.jpg', 1)
nine = ('./week2 - y/people_pictures\\zendaya4.jpg', 4)
ten = ('./week2 - y/people_pictures\\zendaya3.jpg', 4)
eleven = ('./week2 - y/people_pictures\\rihanna1.jpg', 2)
twelve = ('./week2 - y/people_pictures\\zendaya1.jpg', 4)
thirteen = ('./week2 - y/people_pictures\\zendaya2.jpg', 4)
list_of_nodes = [two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen]

# plot the result of the whisper function by category
labels_list = []
ordered_nodes_list = []
for n in list_of_nodes:
    if n[1] in labels_list:
        ordered_nodes_list[labels_list.index(n[1])].append(n)
    else:
        ordered_nodes_list.append([n])
        labels_list.append(n[1])

max_len = 0
for no in ordered_nodes_list:
    if len(no) > max_len:
        max_len = len(no)
        
rows = len(no)
cols = len(ordered_nodes_list)
for y in range(0, cols):
    for i in range(0, len(ordered_nodes_list[y])):
        plt.subplot(rows, cols, i * cols + y + 1)
        image = plt.imread(ordered_nodes_list[y][i][0])
        plt.imshow(image)
        plt.axis('off')
plt.show()
"""