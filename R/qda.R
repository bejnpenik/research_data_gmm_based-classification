source("testing_functions.R")
qda.results <- list()
qda.errors <- c('Abalone', 'DiabetesPic','Digits','Ecoli', 'Glass','Iono','Leaf', 'Statlog', 'SteelPlates', 'Weka', "Image", "Yeast")
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% qda.errors)
        {
        qda.results[[dataset.name]] = NA
    }
    else{
        qda.results[[dataset.name]] = suppressWarnings(qda.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(qda.results)
write(exportJson, "qdaresults.json")