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
color_vector = [
    match species:
        case 'Iris-setosa':
            'r'
        case 'Iris-virginica':
            'g'
        case 'Iris-versicolor':
            'b'
    for species in labels
]
attribute_set = {sepal_length, sepal_width, petal_length, petal_width}
for attribute_x in attribute_set:
    for attribute_y in attribute_set:
        if (attribute_y != attribute_x):
            plt.scatter(attribute_x, attribute_y)
print(sepal_length)

plt.scatter(xs, ys, c=colors)
plt.savefig("plot.png")