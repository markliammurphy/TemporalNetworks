# TemporalNetworks


## Contents
TimeVaryingNetwork.py - This is a class for the generation and visualization of various metrics of networks as they evolve over time. Includes principal eigenvalue, edges, wedges, triangles, spoons, squares, and pentagons. Relies on NetorkX for graph data.
reality_mining_scripts - Contains scripts for extracting and manipulating some of the location data from the MIT Reality Mining project (Eagle, Pentland, and Lazar, 2009). More can be learned about the project and paper [here](http://realitycommons.media.mit.edu/). Since the data is provided in MATLAB, the cleaning is partially in MATLAB and partially in Python.

## Summary
The class TimeVaryingNetwork was built as an exercise in using Python's object-oriented capabilities. It was then applied to the Reality Mining project to generate plots and data from various metrics that describe the nature of a network. The inspiration for this draws from network science, as shown in [Holme and Saramaki's Review article](https://arxiv.org/abs/1108.1780).

Some basic network properties are plotted below.

![text](https://github.com/markliammurphy/TemporalNetworks/images/edges.jpg "Plot of Edges Over Time")
![text](https://github.com/markliammurphy/TemporalNetworks/images/eigenvalues.jpg "Plot of Principle Eigenvalue Over Time")
![text](https://github.com/markliammurphy/TemporalNetworks/images/triangles.jpg "Plot of Triangles Over Time")

