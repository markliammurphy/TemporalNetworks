import csv
from TimeVaryingNetwork import *
import matplotlib.pyplot as plt
import csv


edges = []
times = []

with open('reality_mining_edges.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        edges.append(np.array([int(row['node1']), int(row['node2'])]))
        times.append(np.array([int(row['time1']), int(row['time2'])]))

edges = np.array(edges)
times = np.array(times)
nodes = np.array([x for x in range(106)])

rm60 = TimeVaryingNetwork(interval=60)
rm60.build_network(nodes, edges, times)

rm120 = TimeVaryingNetwork(interval=120)
rm120.build_network(nodes, edges, times)

rm180 = TimeVaryingNetwork(interval=180)
rm180.build_network(nodes, edges, times)

rm240 = TimeVaryingNetwork(interval=240)
rm240.build_network(nodes, edges, times)

rm300 = TimeVaryingNetwork(interval=300)
rm300.build_network(nodes, edges, times)


#plt.hist(rm180.principal_eig(), bins=30)

#rm300.plot_edges()
eigs = rm240.principal_eig()
edges = rm240.number_of_edges()
wedges = rm240.number_of_wedges()
triangles = rm240.number_of_triangles()
spoons = rm240.number_of_spoons()
quadrangles = rm240.number_of_quadrangles()

out = np.array([
    list(range(0, len(eigs))),
    eigs,
    edges,
    wedges,
    triangles,
    spoons,
    quadrangles
])


with open('reality_mining_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([['timestep', 'principle eig', 'edges', 'wedges', 'triangles', 'spoons', 'quadrangles']])
    writer.writerows(out.transpose())