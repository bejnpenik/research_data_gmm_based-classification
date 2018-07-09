source("testing_functions.R")
rebmix.results <- list()
rebmix.errors <-c()
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% rebmix.errors)
        {
        rebmix.results[[dataset.name]] = NA
    }
    else{
    rebmix.results[[dataset.name]] = suppressWarnings(rebmix.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(rebmix.results)
write(exportJson, "rebmixresults.json")