from matplotlib import pyplot as plt
import os
from glob import glob

file_paths = glob(os.path.join("./week2 - y/people_pictures", "*.jpg"))

images = []

for i in range(0, len(file_paths)):
    images.append(plt.imread(file_paths[i]))

rows = 4
cols = 10
pic_num = 0

for y in range(0, cols):
    for i in range(0, rows):
        plt.subplot(rows, cols, i * cols + y + 1)
        plt.imshow(images[pic_num])
        plt.axis('off')
        if pic_num % rows == 0:
            result = str(file_paths[pic_num])
            result = result[28:len(result) - 5]
            plt.title(result)
        
        pic_num += 1
        
plt.show()