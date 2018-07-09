source("testing_functions.R")
rebmvnorm.results <- list()
rebmvnorm.errors <- c()
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% rebmvnorm.errors)
        {
        rebmvnorm.results[[dataset.name]] = NA
    }
    else{
    rebmvnorm.results[[dataset.name]] = suppressWarnings(rebmvnorm.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(rebmvnorm.results)
write(exportJson, "rebmvnormresults.json")