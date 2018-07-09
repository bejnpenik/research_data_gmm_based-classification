source("testing_functions.R")
nnet.results <- list()
nnet.errors <- c("Image","Iono")
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% nnet.errors)
        {
        nnet.results[[dataset.name]] = NA
    }
    else{
        nnet.results[[dataset.name]] = suppressWarnings(nnet.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(nnet.results)
write(exportJson, "nnetresults.json")