import os
import json
import numpy as np
def IQR(dist):
    return np.percentile(dist, 75) - np.percentile(dist, 25)
def CI(dist, alpha=0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(dist, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(dist, p))
    return (lower, upper)
algorithm_file = {
    "rebmvnorm":"REBMVNORM",
    "rebmix":"REBMIX",
    "mda": "MDA",
    "mclust":"MCLUST",
    "lda":"LDA",
    "qda":"QDA",
    "nnet":"NNET",
    "multinom":"LOGREG",
    "svm":"SVM"    
}
all_results = {}

for file in os.listdir("rezultati"):
    if file.endswith(".json"):
        
        file_path = os.path.join("rezultati", file)
        with open(file_path) as f:
            all_results[algorithm_file[file[:file.find("results.json")]]] = json.load(f)
#method_names = all_results.keys()
#database_names = all_results["LDA"].keys()))
result_error_median = {i + " " + j:np.median(all_results[i][j]["error"])
						if all_results[i][j] is not None else None 
						for i in all_results 
						for j in all_results[i]}
result_error_IQR = {i + " " + j:IQR(all_results[i][j]["error"])
						if all_results[i][j] is not None else None 
						for i in all_results 
						for j in all_results[i]}
result_all_time = {i + " " + j:np.sum(all_results[i][j]["time.train"] + all_results[i][j]["time.test"])
						if all_results[i][j] is not None else None 
						for i in all_results 
						for j in all_results[i]}
result_CI_error = {i + " " + j:CI(all_results[i][j]["error"])
						if all_results[i][j] is not None else None 
						for i in all_results 
						for j in all_results[i]}
results = {
	"Median": result_error_median,
	"IQR": result_error_IQR,
	"Time": result_all_time,
	"CI": result_CI_error
}
#print(results)
j = json.dumps(results, indent=4)
f = open('result_processed.json', 'w')
print(j, file=f)
f.close()
