source("testing_functions.R")
mda.results <- list()
mda.errors <- c('Ecoli', 'Iono', 'Leaf', 'SteelPlates', 'Weka')
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% mda.errors)
        {
        mda.results[[dataset.name]] = NA
    }
    else{
        mda.results[[dataset.name]] = suppressWarnings(mda.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(mda.results)
write(exportJson, "mdaresults.json")