import pandas as pd 
from matplotlib import pyplot as plt
import numpy

# load
file_path = "./data/iris.data"
attributes = pd.read_csv(file_path, usecols=[0,1,2,3])
labels = pd.read_csv(file_path, usecols=[4])

print("attributes:")
print(attributes)
print("label (species):")
print(labels)

# graph

# get each attribute vector
sepal_length = numpy.array(attributes)[:,0:1]
sepal_width = numpy.array(attributes)[:,1:2]
petal_length = numpy.array(attributes)[:,2:3]
petal_width = numpy.array(attributes)[:,3:4]

# create a graph using 2 attribute vectors (for non-same attributes) for all possible pairs
# red is Iris-setosa, green is Iris-virginica, blue is Iris-versicolor
color_map = {
    'Iris-setosa': 'red',
    'Iris-virginica': 'green',
    'Iris-versicolor': 'blue'
}
color_vector = [color_map[species] for species in labels]
attribute_arr = [sepal_length, sepal_width, petal_length, petal_width]
attribute_category_strings = ["sepal length", "sepal width", "petal length", "petal width"]

fig, axes = plt.subplots(4,4, constrained_layout=True)
for x_index in range(4):
    for y_index in range(4):
        if (x_index != y_index):
            axes[x_index,y_index].scatter(attribute_arr[x_index], attribute_arr[y_index], c=color_vector)
            axes[x_index,y_index].set_title(f"{attribute_category_strings[y_index]} vs. {attribute_category_strings[x_index]}", fontsize=10)
            
plt.savefig(f"plots.png")


