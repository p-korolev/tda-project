# Topological Data Analysis Project

Repository for a TDA project analyzing Mapper graph structures influenced by correlation in bivariate data.

## Background & Objective

The Mapper algorithm is a process that essentially generates discrete graphs from data sets of finite dimension. In Mapper, the graph's general structure can help identify transition periods, outliers, and similarity between data points through customizable filtration processes.

The Betti values $\beta_0$, $\beta_1$ quantify properties like connectedness and cycles in our graph. For example, a fully connected graph would emit $\beta_0=1$, and a value $\beta_1>0$ tells us that our graph contains one or more cycles.

For a bivariate data set $X = \set{(x_i, y_i)}$ and its Mapper graph $G_X$, we aim to find a relationship between the correlation $\rho(x,y)$ and Betti values $\beta_0(G)$, $\beta_1(G)$. 

## Directory Info

### Helper Files

**exp_f.py**: Main helper module used to build Mapper graphs, generate synthetic data, and run simulations

**/tda_helper/**: Holds helper modules used in secondary and test notebooks

### Simulations and Experiments

**experiments.ipynb**: Main notebook used to generate simulations and plot outputs for paper

**example_tda.ipynb**: Sandbox to test Mapper algorithm on real-world data

### Visuals

**/experiment_outputs/**: Stores core experiment result plots used in paper

**/examples/**: Stores example Mapper graphs generated from miscellaneous data

### Main Paper

**TDA_paper.pdf**: Formal theory and study on the Mapper algorithm, hypotheses, and experimental results
