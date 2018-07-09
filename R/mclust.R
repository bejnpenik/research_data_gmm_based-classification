source("testing_functions.R")
mclust.results <- list()
mclust.errors <- c('Abalone', 'Glass', 'Leaf', 'Skin', 'Statlog', 'WineQualityRed', "WineQualityWhite")
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% mclust.errors)
        {
        mclust.results[[dataset.name]] = NA
    }
    else{
    mclust.results[[dataset.name]] = suppressWarnings(mclust.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(mclust.results)
write(exportJson, "mclustresults.json")