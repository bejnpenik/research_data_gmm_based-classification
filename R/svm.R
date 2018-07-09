source("testing_functions.R")
svm.results <- list()
svm.errors <- c()
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% svm.errors)
        {
        svm.results[[dataset.name]] = NA
    }
    else{
    svm.results[[dataset.name]] = suppressWarnings(svm.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(svm.results)
write(exportJson, "svmresults.json")