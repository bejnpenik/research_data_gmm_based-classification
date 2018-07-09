source("testing_functions.R")
lda.results <- list()
lda.errors <- c("Parkinsons", "Iono", "Image", "Ecoli", "SensorlessDrive")
for (dataset.name in names(datasets)){
    print(dataset.name)
    if (dataset.name %in% qda.errors)
        {
        lda.results[[dataset.name]] = NA
    }
    else{
        lda.results[[dataset.name]] = suppressWarnings(lda.estimates(dataset.name, datasets[[dataset.name]]$data, datasets[[dataset.name]]$class.no, datasets[[dataset.name]]$formula))
        }
    }
require(RJSONIO)
exportJson <- toJSON(lda.results)
write(exportJson, "ldaresults.json")
