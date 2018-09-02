import csv

'''
Cleaning the reality mining data
Read in csv with a column for each node and a row for each minute
Write a csv with a row for every connection
'''


# store completed edges in a list of dicts
edges = []

# adjacency list for currently connected nodes,
# and list for the start times of those connections
adj = {x: [] for x in range(106)}
adj_times = {x: {} for x in range(106)}

with open('locations.csv') as csvfile:
    loc_reader = csv.reader(csvfile, delimiter=',')

    for t, row in enumerate(loc_reader):
        # look at each node, see if it is connected to any other
        # print('time ' + str(t))
        for i, loc_i in enumerate(row):
            if loc_i == '0':
                continue

            # list of indices of connections
            connections = [j for j, loc_j in enumerate(row)
                           if loc_i == loc_j and j != i]
            # update adjacency list for i
            for j in connections:
                if j not in adj[i]:
                    adj[i].append(j)
                    adj_times[i][j] = t
            # output as edge if in adjacency list, but not in connections
            end_connections = [j for j in adj[i] if j not in connections]
            for j in end_connections:
                # only write it once
                if i < j:
                    entry = {'node1': i,
                             'node2': j,
                             'time1': adj_times[i][j],
                             'time2': t}
                    edges.append(entry)
                # js times
                adj_times[i].pop(j)
                adj[i].remove(j)

# write to CSV
with open('reality_mining_edges.csv', 'w') as csvfile:
    fieldnames = ['node1', 'node2', 'time1', 'time2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in edges:
        writer.writerow(entry)
