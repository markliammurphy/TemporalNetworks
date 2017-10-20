import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import math


def time_plot(func):
    '''
    Decorator function for plotting global attributes over time
    :param func: plotting function
    :return: modified plotting function
    '''
    def new_plotter(self, weighted=False):
        fig, ax = func(self, weighted=weighted)
        plt.xlabel("timestep")
        ax.set_facecolor('#F5F5F5')
        plt.show()
    return new_plotter

class TimeVaryingNetwork:

    def __init__(self, interval=1440):
        self.interval = interval
        self.graphs = None
        self.unweighted_graphs = None
        self.N = 0
        self.subgraph_funcs = {
            'edges': self.number_of_edges,
            'wedges': self.number_of_wedges,
            'triangles': self.number_of_triangles,
            'spoons': self.number_of_spoons,
            'quadrangles': self.number_of_quadrangles,
            'pentagons': self.number_of_pentagons
        }

    def __hash__(self):
        return sum([g.__hash__() for g in self.graphs])

    def build_network(self, nodes, edges, times):
        '''
        Construct a time varying network
        :param nodes: numpy array containing the unique names
        of all nodes in the network
        :param edges: e by 2 numpy matrix specifying the nodes
        for each of e edges
        :param times: e by 2 numpy matrix specifying the times
        for which each edge is alive
        :return: None
        '''
        network_iter = zip([(t1, t2) for (t1, t2) in times],
                           [(e1, e2) for (e1, e2) in edges])
        lst = np.array([[t, e] for t, e in network_iter])

        # rename nodes if we have string names
        labels = {}
        numbers = {}
        node_num = itertools.count()
        numbered_edges = []

        for n in nodes:
            num = node_num.__next__()
            labels[num] = n
            numbers[n] = num

        for n1, n2 in edges:
            numbered_edges.append([numbers[n1], numbers[n2]])

        network_iter = zip([(t1, t2) for (t1, t2) in times],
                           [(e1, e2) for (e1, e2) in numbered_edges])
        data = np.array(list(network_iter))

        # max timestep
        T = self.interval
        N = int(round(max(data[:, 0, 1]) / T))

        # put every substrate graph in graphs
        graphs = [nx.Graph() for n in range(N)]
        unweighted_graphs = [nx.Graph() for n in range(N)]
        for n in range(N):
            start = n * T
            end = (n + 1) * T

            # get all edges for times ending after
            # the start time and starting before the end time
            subset = data[np.logical_and(data[:, 0, 0] < end,
                                         data[:, 0, 1] > start)]

            # find list of weights
            weights = [(min(t2, end) - max(t1, start)) / T
                       for (t1, t2), nodes in subset]

            node_iter = zip([labels[x] for x in subset[:, 1, 0]],
                            [labels[x] for x in subset[:, 1, 1]],
                            weights)
            uw_iter = zip([labels[x] for x in subset[:, 1, 0]],
                          [labels[x] for x in subset[:, 1, 1]])

            # for node in node_iter:
            graphs[n].add_weighted_edges_from(list(node_iter))
            unweighted_graphs[n].add_edges_from(list(uw_iter))

        self.graphs = graphs
        self.unweighted_graphs = unweighted_graphs
        self.N = N
        return None

    def substrate_graph(self, n):
        '''
        Get substrate graph at nth time interval
        :param n: time interval
        :return: a networkx Graph object
        '''
        if n > self.N-1 or n < 0:
            raise Exception("Time interval index must be between 0 and " +
                            str(self.N-1))
        return self.graphs[n]

    def get_graphs(self, n, weighted=False):
        '''
        Gets graphs at
        :param n: Indices of graphs to get
        :param weighted: True uses weighted graph, False by default
        :return: List of networkx objects
        '''
        if n is None:
            graphlist = self.graphs if weighted else self.unweighted_graphs
        else:
            graphlist = [self.graphs[x]
                         if weighted else self.unweighted_graphs[x]
                         for x in (n if isinstance(n, list) else [n])]
        return graphlist

    def eigenvalues(self, n=None, weighted=False):
        '''
        Get the eigenvalues of network
        :param n: the time period(s) for which to get the eigenvalues
        :param weighted: True uses weighted graph, False by default
        :return: an array of numpy array with a column of eigenvalues
        for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return np.array([nx.adjacency_spectrum(g)
                         if nx.number_of_edges(g) > 0 else np.array([])
                         for g in graphlist])

    def principal_eig(self, n=None, weighted=False):
        '''
        Ger principle eigenvalues of network
        :param n: time period(s) for which to get the principle eigenvalues
        :param weighted: True uses weighted graph, False by default
        :return:  numpy array of eigenvalues
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return np.array([max(np.real(nx.adjacency_spectrum(g)))
                         if nx.number_of_edges(g) > 0 else None
                         for g in graphlist])

    def number_of_edges(self, n=None, weighted=False):
        '''
        Get the number of edges in the network, where an edge is a distinct
        connection between two nodes
        :param n: the time period(s) for which to get the number of edges
        :param weighted: True uses weighted graph, False by default
        :return: a numpy array of the number of nodes for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return np.array([nx.number_of_edges(g) for g in graphlist])

    def number_of_triangles(self, n=None, weighted=False):
        '''
        Get the number of triangles in the network, where a triangle is 
        defined by three connected nodes
        :param n: the time period(s) for which to get the number of triangles
        :param weighted: True uses weighted graph, False by default
        :return: a numpy array of the number of triangles for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return np.array(
            [sum(sparse.csr_matrix.diagonal(
                nx.adjacency_matrix(g).__pow__(3)))/6
             if nx.number_of_edges(g) > 0 else 0
             for g in graphlist])

    def number_of_wedges(self, n=None, weighted=False):
        '''
        Get the number of wedges in the network, where a wedge is a triangle
        without one of its edges
        :param n: the time period(s) for which to get the number of wedges
        :param weighted: True uses weighted graph, False by default
        :return: a numpy array of the number of wedges for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return np.array([sum([math.factorial(d)/(2*math.factorial(d-2))
                              if d > 1 else 0 for d in g.degree().values()])
                         if nx.number_of_edges(g) > 0 else 0
                         for g in graphlist])

    def number_of_quadrangles(self, n=None, weighted=False):
        '''
        Get the number of quadrangles in the network
        :param n: the time period(s) for which to get the number of quadrangles
        :param weighted:  True uses weighted graph, False by default
        :return: a numpy array of the number of quadrangles for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        tr_adj_4 = np.array([sum(sparse.csr_matrix.diagonal(nx.adjacency_matrix(g).__pow__(4)))
                            if nx.number_of_edges(g) > 0 else 0 for g in graphlist])
        wedges = self.number_of_wedges(n, weighted=weighted)
        edges = self.number_of_edges(n, weighted=weighted)
        return (1/8)*tr_adj_4 - (1/2)*wedges - (1/4)*edges

    def number_of_spoons(self, n=None, weighted=False):
        '''
        Get the number of spoons in the network, where a spoon is a triangle
        with a connection to a fourth node from one vertex
        :param n: the time period(s) for which to get the number of spoons
        :param weighted: True uses weighted graph, False by default
        :return: a numpy array of the number of spoons for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        return [np.dot(sparse.csr_matrix.diagonal(nx.adjacency_matrix(g).__pow__(3))/2,
                       np.array(list(g.degree().values())))
                if nx.number_of_edges(g) > 0 else 0 for g in graphlist]

    def number_of_pentagons(self, n=None, weighted=False):
        '''
        Get the number of pentagons in the network
        :param n: the time period(s) for which to get the number of quadrangles
        :param weighted:  True uses weighted graph, False by default
        :return: a numpy array of the number of quadrangles for each time period
        '''
        graphlist = self.get_graphs(n, weighted=weighted)
        tr_adj_5 = np.array([sum(sparse.csr_matrix.diagonal(nx.adjacency_matrix(g).__pow__(5)))
                             if nx.number_of_edges(g) > 0 else 0 for g in graphlist])
        triangles = self.number_of_triangles(n, weighted=weighted)
        spoons = self.number_of_spoons(n, weighted=weighted)
        return (1 / 10) * tr_adj_5 - spoons - 3 * triangles

    @time_plot
    def plot_principal_eig(self, n=None, weighted=False):
        '''
        Plot the principal eigenvalue over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        ax.plot(np.real(self.principal_eig(n=n, weighted=weighted)))
        plt.ylabel("principle eigenvalue")
        plt.title("Principle Eigenvalue Over Time")
        return fig, ax

    @time_plot
    def plot_edges(self, n=None, weighted=False):
        '''
        Plot number of edges over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_edges(n=n, weighted=weighted))
        plt.ylabel("number of edges")
        plt.title("Number of Edges Over Time")
        return fig, ax

    @time_plot
    def plot_triangles(self, n=None, weighted=False):
        '''
        Plot number of triangles over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_triangles(n=n, weighted=weighted))
        plt.ylabel("number of triangles")
        plt.title("Number of Triangles Over Time")
        return fig, ax

    @time_plot
    def plot_wedges(self, n=None, weighted=False):
        '''
        Plot number of triangles over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_wedges(n=n, weighted=weighted))
        plt.ylabel("number of wedges")
        plt.title("Number of Wedges Over Time")
        return fig, ax

    @time_plot
    def plot_quadrangles(self, n=None, weighted=False):
        '''
        Plot number of quadrangles over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_quadrangles(n=n, weighted=weighted))
        plt.ylabel("number of quadrangles")
        plt.title("Number of Quadrangles Over Time")
        return fig, ax

    @time_plot
    def plot_spoons(self, n=None, weighted=False):
        '''
        Plot number of spoons over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_spoons(n=n, weighted=weighted))
        plt.ylabel("number of spoons")
        plt.title("Number of Spoons Over Time")
        return fig, ax

    @time_plot
    def plot_pentagons(self, n=None, weighted=False):
        '''
        Plot number of pentagons over time
        :param n: the time period(s) for which to plot
        :param weighted: True uses weighted graph, False by default
        :return: figure and axis objects
        '''
        fig, ax = plt.subplots()
        plt.plot(self.number_of_pentagons(n=n, weighted=weighted))
        plt.ylabel("number of pentagons")
        plt.title("Number of Pentagons Over Time")
        return fig, ax

    def plot_substrate(self, n):
        '''
        Plot the subtrate graph at time interval n using matplotlib
        :param n: the time period for which to plot the substrate graph
        :return: None
        '''
        nx.draw_networkx(self.substrate_graph(n))
        plt.show()
        return None

    def plot_subgraphs2D(self, subgraph1, subgraph2, weighted=False):
        '''
        Plots 2D image of evolution of subgraphs
        {'edges', 'wedges', 'triangles', 'spoons', 'quadrangles', 'pentagons'}
        :param subgraph1: x axis subgraph
        :param subgraph2: y axis subgraph
        :param weighted: True uses weighted graph, False by default
        :return: None
        '''
        x = self.subgraph_funcs[subgraph1](weighted=weighted)
        y = self.subgraph_funcs[subgraph2](weighted=weighted)
        plt.plot(x, y, ':')
        plt.xlabel('number of ' + subgraph1)
        plt.ylabel('number of ' + subgraph2)
        plt.title('Evolution of Subgraphs')
        plt.show()
        return None

    def plot_subgraphs3D(self, subgraph1, subgraph2, subgraph3,
                         weighted=False):
        '''
        Plots 3D image of the desired subgraph
        {'edges', 'wedges', 'triangles', 'spoons', 'quadrangles', 'pentagons'}
        :param subgraph1: x axis subgraph
        :param subgraph2: y axis subgraph
        :param subgraph3: z axis subgraph
        :param weighted: True uses weighted graph, False by default
        :return: None
        '''
        x = self.subgraph_funcs[subgraph1](weighted=weighted)
        y = self.subgraph_funcs[subgraph2](weighted=weighted)
        z = self.subgraph_funcs[subgraph3](weighted=weighted)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(x, y, z, ':')
        ax.set_xlabel('number of ' + subgraph1)
        ax.set_ylabel('number of ' + subgraph2)
        ax.set_zlabel('number of ' + subgraph3)
        plt.show()
        return None
