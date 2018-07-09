import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress
import numpy as np
class Colors:
    RED = "#752a2a"
    BLUE = "#1a97c3"
    ORANGE = "#e86c24"
    YELLOW = "#e9a916"
    GREEN = "#63ba1a"
    CYAN = "#1ac3a2"
    PURPLE = "#551ac3"
    MAGENTA = "#c31a9a"
    DARK = "#696969"
    BLACK = "#000000"
    
class ClusterAnalysis:
    
    COLORS = {
        "REBMVNORM":Colors.RED,
        "REBMIX": Colors.BLUE,
        "MDA": Colors.ORANGE,
        "MCLUST": Colors.GREEN,
        "LDA": Colors.CYAN,
        "QDA": Colors.PURPLE,
        "NNET": Colors.MAGENTA,
        "LOGREG": Colors.DARK,
        "SVM": Colors.YELLOW,
    }
    #TODO: find out what output of linkage means!
    def __init__(self, median, iqr, time, names, colors, method="complete", metric = "euclidean"):
        plt.close("all")
        self._median = np.asarray(median)
        self._iqr = np.asarray(iqr)
        self._time = np.asarray(time)
        self._names = np.asarray(names)
        self._colors = np.asarray(colors)
        self._cluster_tree = linkage(np.array([time, 
        										iqr,
        										median]).T, method = method, metric = metric)
        #fcluster(test_value_complete, t = 10, criterion='maxclust')
        self._clusters = fcluster(self._cluster_tree, t = 10, criterion='maxclust') 
        self._canvas = plt.figure(figsize = (10, 10))
        self._denoised = False
    def recluster(self, t, criterion):
        self._clusters = fcluster(self._cluster_tree, t = t, criterion=criterion)
    def plot_dendogram(self, p=100, truncate_mode="lastp"):
       	_canvas = plt.figure(figsize = (10, 10))
        ax = _canvas.add_subplot(111)
        dendrogram(self._cluster_tree, p=p, truncate_mode=truncate_mode, ax=ax, no_labels=True)
        ax.set_ylabel("Distance")
        #plt.show()
    def plot_data_colored(self):
        _canvas = plt.figure(figsize = (10, 10))
        #_canvas.suptitle("Standardized results of medians, interquartile ranges and computaion time for all methods on all results", fontsize=16)
        ax = _canvas.add_subplot(221, projection='3d')
        ax.scatter(self._time,self._iqr, self._median, c = self._colors)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        ax.set_zlabel("Median")
        ax = _canvas.add_subplot(222)
        ax.scatter(self._time, self._iqr, c = self._colors)#, c = self._clusters)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        ax = _canvas.add_subplot(223)
        ax.scatter(self._time, self._median, c = self._colors)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        ax = _canvas.add_subplot(224)
        ax.scatter(self._iqr, self._median, c = self._colors)#, c = self._clusters)
        ax.set_xlabel("Interquartile range")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        #plt.show()
    def plot_data(self, recluster = None):
        if recluster:
            self.recluster(recluster["t"], recluster["criterion"])
        _canvas = plt.figure(figsize = (10, 10))
        #_canvas.suptitle("Standardized results of medians, interquartile ranges and computaion time for all methods on all results", fontsize=16)
        ax = _canvas.add_subplot(221, projection='3d')
        ax.scatter(self._time,self._iqr, self._median)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        ax.set_zlabel("Median")
        ax = _canvas.add_subplot(222)
        ax.scatter(self._time, self._iqr)#, c = self._clusters)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        ax = _canvas.add_subplot(223)
        ax.scatter(self._time, self._median)#, c = self._clusters)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        slope, intercept, r_value, p_value, std_err = linregress(self._iqr, self._median)
        ax = _canvas.add_subplot(224)
        ax.scatter(self._iqr, self._median)#, c = self._clusters)
        ax.plot(self._iqr, intercept + slope*self._iqr, 'r', label='fitted line')
        ax.text(0.75, -2.5, "Regression line\nwith correlation\ncoef. R = 0.33", color="r")
        ax.set_xlabel("Interquartile range")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        #plt.show()
    def plot_clusters(self, recluster = None, colors = None):
        if recluster:
            self.recluster(recluster["t"], recluster["criterion"])
        _canvas = plt.figure(figsize = (10, 10))
        #_canvas.suptitle("Standardized results of medians, interquartile ranges and computaion time for all methods on all results", fontsize=16)
        ax = _canvas.add_subplot(221, projection='3d')
        _colors = self._clusters
        if colors:
        	_colors = [colors[_] for _ in self._clusters]
        ax.scatter(self._time ,self._iqr, self._median, c = _colors)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        ax.set_zlabel("Median")


        ax = _canvas.add_subplot(222)
        ax.scatter(self._time, self._iqr, c = _colors)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Interquartile range")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        ax = _canvas.add_subplot(223)
        ax.scatter(self._time, self._median, c = _colors)
        ax.set_xlabel("LogTime")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        ax = _canvas.add_subplot(224)
        ax.scatter(self._iqr, self._median, c = _colors)
        ax.set_xlabel("Interquartile range")
        ax.set_ylabel("Median values")
        #for i in np.unique(self._clusters):
        #    x, y = np.mean(self._data[self._clusters == i], axis=0)
        #    ax.text(x,y,str(i), fontsize=15)
        #plt.show()
    def algorithm_prior(self, cluster):
        _cluster_counts = {_u:_c for _u,_c in zip(*np.unique(np.array(self._names)[self._clusters==cluster], return_counts = True))}
        _all_counts = {_u:_c for _u,_c in zip(*np.unique(self._names, return_counts = True))}
        return {alg:_cluster_counts[alg]/_all_counts[alg] if alg in _cluster_counts.keys() else 0 for alg in np.unique(self._names)}
    
    def cluster_prior(self, cluster):
         _cluster_counts = {_u:_c for _u,_c in zip(*np.unique(np.array(self._names)[self._clusters==cluster], 
                                                              return_counts = True))}
         _cluster_holds = np.sum(self._clusters==cluster)
         return {alg:_cluster_counts[alg]/_cluster_holds if alg in _cluster_counts.keys() else 0 for alg in np.unique(self._names)}
    def algorithm_conditional(self, algorithm):
        _algorithm_counts = {_u:_c for _u,_c in zip(*np.unique(self._clusters[self._names==algorithm],
                                                              return_counts=True))}
        _algorithm_holds = np.sum(self._names==algorithm)
        return {cluster:_algorithm_counts[cluster]/_algorithm_holds if cluster in _algorithm_counts.keys() else 0 for cluster in np.unique(self._clusters)}


    def max_posteriors_cluster(self, cluster):
        _prior = self.cluster_prior(cluster)
        _a_prior = self.algorithm_prior(cluster)
        _max = 0
        _a_max = None
        for alg in _prior:
            _ = _prior[alg]*_a_prior[alg]
            if _ > _max:
                _max = _
                _a_max = alg
        return _a_max
    def max_posteriors_clusters(self):
        clusters = {}
        for cluster in np.unique(self._clusters):
            clusters[cluster] = self.max_posteriors_cluster(cluster)
        return clusters
    def max_posterior_algorithm(self, algorithm):
        clusters = {}
        for cluster in np.unique(self._clusters):
            _prior = self.cluster_prior(cluster)
            _a_prior = self.algorithm_prior(cluster)
            clusters[cluster] = _prior[algorithm]*_a_prior[algorithm]
        _max = 0
        _c_max = None
        for cluster in clusters:
            if clusters[cluster] > _max:
                _max = clusters[cluster]
                _c_max = cluster
        return _c_max
    def max_posterior_algorithms(self):
        algorithms = {}
        for alg in np.unique(self._names):
            algorithms[alg] = self.max_posterior_algorithm(alg)
        return algorithms
       
    
    def get_cluster_info(self, cluster):
        if cluster not in  np.unique(self._clusters):
            raise Exception("Cluster %s is not valid!")%str(cluster)
        _cluster_data = self._data[self._clusters == cluster]
        _mean = np.mean(_cluster_data, axis=0)
        _std = np.cov(_cluster_data.T)
        _priors = self.cluster_prior(cluster)
        _a_priors = self.algorithm_prior(cluster)
        _procents = len(_cluster_data)/len(self._data)
        _algorithms_in_cluster = " ,".join(np.unique(self._names[self._clusters == cluster]))
        print(""" Info about cluster %d"""%cluster)
        print("Cluster holds %s"%str(_procents*100) + "% of data")
        print("""Mean of cluster is:""")
        print(_mean)
        print("Covariance of cluster is:")
        print(_std)
        print("Present algorithms in cluster are: %s with following sizes:"%_algorithms_in_cluster)
        for alg in _priors:
            if _priors[alg] != 0:
                print("    Algorithm %s contains %s"%(alg, str(100*_priors[alg])) + " % of the cluster size")
                print("    Additionally, algorithm %s has %s"%(alg, str(100*_a_priors[alg])) + "% of its own size")
    def cluster_analysis(self, truncate = False):
        if not truncate:
            for cluster in np.unique(self._clusters):
                self.get_cluster_info(cluster)
        _best_clusters = self.max_posteriors_clusters()
        for cluster in _best_clusters:
            print("For cluster %s best algorithm is %s"%(str(cluster), _best_clusters[cluster]))
        _best_algorithms = self.max_posterior_algorithms()
        for alg in _best_algorithms:
            print("For algorithm %s best cluster is %s"%(alg, str(_best_algorithms[alg])))
    def clusters(self):
        return {_:self.cluster_prior(_) for _ in np.unique(self._clusters)}
    def algorithms(self):
        return {_:self.algorithm_prior(_) for _ in np.unique(self._clusters)}
    def get_cluster_table(self):
        _best_clusters = {_:self.cluster_prior(_) for _ in np.unique(self._clusters)}
        _best_algorithms = {_:self.algorithm_prior(_) for _ in np.unique(self._clusters)}
        _full_output = {}
        for cluster in _best_clusters: 
            _full_output[cluster] = {}
            for algorithm in _best_clusters[cluster]:
                #print(cluster, algorithm)
                _full_output[cluster][algorithm] = _best_algorithms[cluster][algorithm]*_best_clusters[cluster][algorithm]
        return _full_output
    def cluster_table(self):
        _table_dict = self.get_cluster_table()
        print("\t"+"\t".join(range(np.unique(self._names))))
        for cluster in _table_dict:
            print(cluster+"\t"+"\t".join(_table_dict.values()))
    def posterior_cluster(self, cluster):
        _cluster_conditional = self.cluster_prior(cluster)
        _algorithm_prior = {_:sum(self._names==_)/len(self._names) for _ in np.unique(self._names)}
        _posterior = {}
        _prob = np.sum(np.array(list(_cluster_conditional.values()))*np.array(list(_algorithm_prior.values())))
        for alg in _cluster_conditional:
            #print(_cluster_conditional[alg])
            #print(_algorithm_prior[alg])
            _posterior[alg] = _cluster_conditional[alg]*_algorithm_prior[alg]/_prob
        return _posterior
    def posterior_clusters(self):
        all_posteriors = {}
        for cluster in np.unique(self._clusters):
           all_posteriors[cluster] = self.posterior_cluster(cluster)
        return all_posteriors
    def posterior_algorithm(self, algorithm):
        _algorithm_conditional = self.algorithm_conditional(algorithm)
        #print(_algorithm_conditional)
        _cluster_prior = {_:sum(self._clusters==_)/len(self._clusters) for _ in np.unique(self._clusters)}
        _posterior = {}
        #print(len(_algorithm_conditional.values()))
        #print(len(_cluster_prior.values()))
        _prob = np.sum(np.array(list(_algorithm_conditional.values()))*np.array(list(_cluster_prior.values())))
        for cluster in _algorithm_conditional:
            #print(cluster)
            #print(_cluster_conditional[alg])
            #print(_algorithm_prior[alg])
            _posterior[cluster] = _algorithm_conditional[cluster]*_cluster_prior[cluster]/_prob
        return _posterior
    def posterior_algorithms(self):
        all_posteriors = {}
        for algorithm in np.unique(self._names):
            all_posteriors[algorithm] = self.posterior_algorithm(algorithm)
        return all_posteriors
