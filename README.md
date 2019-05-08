# Research data for the article: Review and comparison of the Gaussian mixture modeling for the classification tasks

All data used for the analysis and comparisons in the article is available in this git repository.
For the estimation purposes of classification errors and computation times 5-fold cross validation is used. Estimation of values of classification errors for different datasets was made in the R programming language. All belonging code is available in an R / subdirectory. All datasets used for the study are available additionally, yet they are all downloaded from the UCI Machine learning repository. Datasets are available at datasets/ subdirectory. All python code used for clustering and additional comparison of classification methods is available (result_processing.py is the script used for conversion of all results to medians, interquartile ranges and total computation time, and cluster_analysis.py is set of functions used for cluster analysis and plotting). Analysis.ipynb is a juypter notebook used for examples of usage of cluster_analysis.py functions.
