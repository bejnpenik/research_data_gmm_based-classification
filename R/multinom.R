source("testing_functions.R")
multinom.results <- list()
multinom.errors <- c()
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% multinom.errors)
        {
        multinom.results[[dataset.name]] = NA
    }
    else{
    multinom.results[[dataset.name]] = suppressWarnings(multinom.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(multinom.results)
write(exportJson, "multinomresults.json")