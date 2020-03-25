import networkx as nx
from pandas import read_csv, Series
from numpy import array
from collections import namedtuple

DataSet = namedtuple(
    'DataSet',
    field_names=['X_train', 'y_train', 'X_test', 'y_test', 'network']
)


g = nx.karate_club_graph()

attributes = read_csv('data/karate.attributes.csv',
                      index_col=['node'])

for attribute in attributes.columns.values:
    nx.set_node_attributes(
        g, values=Series(attributes[attribute],
                         index=attributes.index).to_dict(),
        name=attribute)

X_train, y_train = map(array, zip(*[
    ([node], data['role'] == 'Administrator')
    for node, data in g.nodes(data=True)
    if data['role'] in {'Administrator', 'Instructor'}
]))
X_test, y_test = map(array, zip(*[
    ([node], data['community'] == 'Administrator')
    for node, data in g.nodes(data=True)
    if data['role'] == 'Member'
]))

res = DataSet(X_train, y_train,
              X_test, y_test, g)

print(res.X_train)
print(res.y_train)