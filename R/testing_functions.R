#set working directory
create.numeric.class <- function(Dataset, class.no){
    n.data <- Dataset
    n.data.labels <- unique(Dataset[, class.no])
    j <- 1
    for (i in n.data.labels){
        #print()
        n.data[Dataset[,class.no]==i,class.no] <- j
        j <- j + 1
    }
    return(as.data.frame(n.data))
}
delete.classes <- function(Dataset, class.no, classes){
    for (d.class in classes){
        Dataset[Dataset[,class.no]==d.class,class.no] <- NA
        Dataset <- Dataset[complete.cases(Dataset),]
    }
    return(Dataset)
}
iris <- read.csv("BazePodatkov/Iris.csv", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
iris <- iris[, -1]
iris.class <- 5
iris.formula <- Species~.
wine <- read.csv("BazePodatkov/wine.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
wine.class <- 1
wine.formula <- V1~.
parkinsons <- read.csv("BazePodatkov/parkinsons.csv", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
parkinsons<- parkinsons[, -c(1)]
parkinsons.class <- 17
parkinsons.formula <- status~.
thyroid <- read.csv("BazePodatkov/thy.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
thyroid.class <- 6
thyroid.formula <- V6~.
sonar <- read.csv("BazePodatkov/sonar.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
sonar.class <- 61
sonar.formula <- V61~.
sonar <- create.numeric.class(sonar, sonar.class)
seeds <- read.csv("BazePodatkov/seeds.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
seeds.class <- 8
seeds.formula <- V8~.
glass <- read.csv("BazePodatkov/glass.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
glass <- glass[, -1]
glass.class <- 10
glass.formula <- V11~.
weka <- read.csv("BazePodatkov/weka_2C.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
weka.class <- 7
weka.formula <- V7~.
weka <- create.numeric.class(weka, weka.class)
ecoli <- read.csv("BazePodatkov/ecoli.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
ecoli <- ecoli[, -c(3,4)]
ecoli.class <- 6
ecoli.formula <- V8~.
ecoli <- create.numeric.class(ecoli, ecoli.class)
leaf <- read.csv("BazePodatkov/leaf.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
leaf <- leaf[,-2]
leaf.class <- 1
leaf <- create.numeric.class(leaf, leaf.class)
leaf.formula <- V1~.
liver <- read.csv("BazePodatkov/liver.data", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
liver <-  liver[, -c(1)]
liver.class <- 6
liver.formula <- class~.
iono <- read.csv("BazePodatkov/iono.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
iono.class <- 35
iono.formula <- V35~.
iono <- create.numeric.class(iono, iono.class)
wisc <- read.csv("BazePodatkov/wisc.csv", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
wisc.class <- 10
wisc.formula <- Class~.
wisc <- create.numeric.class(wisc, wisc.class)
diabetes <- read.csv("BazePodatkov/diabetes.data", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
diabetes.class <- 9
diabetes.formula <- V9~.
vehicle <- read.csv("BazePodatkov/vehicle.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
vehicle.class <- 19
vehicle.formula <- V19~.
vehicle <- create.numeric.class(vehicle, vehicle.class)
vowel <- read.csv("BazePodatkov/vowel.data", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
vowel <-  vowel[, -c(1,2,3)]
vowel.class <- 11
vowel.formula <- V14~.
yeast <- read.csv("BazePodatkov/yeast.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
yeast <- yeast[, -c(1)]
yeast <- yeast[complete.cases(yeast),]
yeast.class <- 9 
yeast.formula <- V10~.
yeast <- create.numeric.class(yeast, yeast.class)
yeast <- delete.classes(yeast, yeast.class, c(10))
steel.plates <- read.csv("BazePodatkov/steel_faults.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
steel.plates.class <- 28
steel.plates.formula <- V28~.
wifi <- read.csv("BazePodatkov/wifi_strength.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
wifi.class <- 8
wifi.formula <- V8~.
abalone <- read.csv("BazePodatkov/abalone.csv", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
abalone <- abalone[, -1]
abalone.class <- 8
abalone.formula <- Razred~.
abalone <- delete.classes(abalone, abalone.class, c(1,2, 24, 25,26,27,28,29))
wilt <- read.csv("BazePodatkov/wilt.txt", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
wilt.class <- 1
wilt.formula <- class~.
wilt <- create.numeric.class(wilt, wilt.class)
landsat <- read.csv("BazePodatkov/landsat.txt", header=FALSE, sep = " " ,stringsAsFactors=FALSE)
landsat.class <- 37
landsat.formula <- V37~.
wine.quality.red <- read.csv("BazePodatkov/winequality-red.csv", header=TRUE, sep = ";" ,stringsAsFactors=FALSE)
wine.quality.red.class <- 12
wine.quality.red.formula <- quality~.
wine.quality.red <- delete.classes(wine.quality.red, wine.quality.red.class, c(3,8))
wine.quality.white <- read.csv("BazePodatkov/winequality-white.csv", header=TRUE, sep = ";" ,stringsAsFactors=FALSE)
wine.quality.white.class <- 12
wine.quality.white.formula <- quality~.
wine.quality.white <- delete.classes(wine.quality.white, wine.quality.white.class, c(3,9))
phase.gesture <- read.csv("BazePodatkov/phase-gesture.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
phase.gesture.class <- 33
phase.gesture.formula <- V33~.
phase.gesture <- create.numeric.class(phase.gesture, phase.gesture.class)
digits <- read.csv("BazePodatkov/pendigits.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
digits.class <- 17
digits.formula <- V17~.
eeg.eye <- read.csv("BazePodatkov/eeg-eye.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
eeg.eye.class <- 15
eeg.eye.formula <- V15~.
htru <- read.csv("BazePodatkov/HTRU.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
htru.class <- 9
htru.formula <- V9~.
telescope <- read.csv("BazePodatkov/telescope.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
telescope.class <- 11
telescope.formula <- V11~.
telescope <- create.numeric.class(telescope, telescope.class)
letter.recognition <- read.csv("BazePodatkov/letterdata.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
letter.recognition.class <- 1
letter.recognition.formula <- V1~.
letter.recognition <- create.numeric.class(letter.recognition, letter.recognition.class)
statlog <- read.csv("BazePodatkov/statlogdata", header=FALSE, sep = " " ,stringsAsFactors=FALSE)
statlog.class <- 10
statlog.formula <-  V10~.
sensorless.drive <- read.csv("BazePodatkov/selfdrivingcar.txt", header=FALSE, sep = " " ,stringsAsFactors=FALSE)
sensorless.drive.class <- 49
sensorless.drive.formula <- V49~.
skin <- read.csv("BazePodatkov/skin_dataset.txt", header=FALSE, sep = "	" ,stringsAsFactors=FALSE)
skin.class <- 4
skin.formula <- V4~.
diabetes.pic <- read.csv("BazePodatkov/DiabeticRetinopathyDebrecen.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
diabetes.pic <- diabetes.pic[complete.cases(diabetes.pic), ]
diabetes.pic.class <- 20
diabetes.pic.formula <- V20~.
indian.liver <- read.csv("BazePodatkov/indian_liver.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
indian.liver <- indian.liver[,-c(2)]
indian.liver <- indian.liver[complete.cases(indian.liver),]
indian.liver.class <- 10
indian.liver.formula <- V11~.
drug.consumption <- read.csv("BazePodatkov/drug_consumption.txt", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
drug.consumption <- drug.consumption[, -c(14:18)]
drug.consumption <- drug.consumption[, -c(15:27)]
drug.consumption.class <- 14
drug.consumption.formula <- V19~.
drug.consumption <- create.numeric.class(drug.consumption, drug.consumption.class)
musk <- read.csv("BazePodatkov/musk.data", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
musk <- musk[, -c(1,2)]
musk.class <- 167
musk.formula <- V169~.
image <- read.csv("BazePodatkov/image.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
image.class <- 1
image.formula <- V1~.
image <- create.numeric.class(image, image.class)
frogs <- read.csv("BazePodatkov/frogsdata.csv", header=TRUE, sep = "," ,stringsAsFactors=FALSE)
frogs <- frogs[, -1]
frogs.class <- 22
frogs.formula <- class~.
banknotes <- read.csv("BazePodatkov/banknote.csv", header=FALSE, sep = "," ,stringsAsFactors=FALSE)
banknotes.class <- 5
banknotes.formula <- V5~.
datasets <- list()
datasets[["Iris"]] = list(data=iris, class.no=iris.class, formula=iris.formula)
datasets[["Wine"]] = list(data=wine, class.no=wine.class, formula=wine.formula)
datasets[["Parkinsons"]] = list(data=parkinsons, class.no=parkinsons.class, formula=parkinsons.formula)
datasets[["Thyroid"]] = list(data=thyroid, class.no=thyroid.class, formula=thyroid.formula)
datasets[["Sonar"]] = list(data=sonar, class.no=sonar.class, formula=sonar.formula)
datasets[["Seeds"]] = list(data=seeds, class.no=seeds.class, formula=seeds.formula)
datasets[["Glass"]] = list(data=glass, class.no=glass.class, formula=glass.formula)
datasets[["Weka"]] = list(data=weka, class.no=weka.class, formula=weka.formula)
datasets[["Ecoli"]] = list(data=ecoli, class.no=ecoli.class, formula=ecoli.formula)
datasets[["Leaf"]] = list(data=leaf, class.no=leaf.class, formula=leaf.formula)
datasets[["Liver"]] = list(data=liver, class.no=liver.class, formula=liver.formula)
datasets[["Iono"]] = list(data=iono, class.no=iono.class, formula=iono.formula)
datasets[["Wisc"]] = list(data=wisc, class.no=wisc.class, formula=wisc.formula)
datasets[["Diabetes"]] = list(data=diabetes, class.no=diabetes.class, formula=diabetes.formula)
datasets[["Vehicle"]] = list(data=vehicle, class.no=vehicle.class, formula=vehicle.formula)
datasets[["Vowel"]] = list(data=vowel, class.no=vowel.class, formula=vowel.formula)
datasets[["Yeast"]] = list(data=yeast, class.no=yeast.class, formula=yeast.formula)
datasets[["SteelPlates"]] = list(data=steel.plates, class.no=steel.plates.class, formula=steel.plates.formula)
datasets[["Wifi"]] = list(data=wifi, class.no=wifi.class, formula=wifi.formula)
datasets[["Abalone"]] = list(data=abalone, class.no=abalone.class, formula=abalone.formula)
datasets[["Wilt"]] = list(data=wilt, class.no=wilt.class, formula=wilt.formula)
datasets[["Landsat"]] = list(data=landsat, class.no=landsat.class, formula=landsat.formula)
datasets[["WineQualityRed"]] = list(data=wine.quality.red, class.no=wine.quality.red.class, formula=wine.quality.red.formula)
datasets[["WineQualityWhite"]] = list(data=wine.quality.white, class.no=wine.quality.white.class, formula=wine.quality.white.formula)
datasets[["PhaseGesture"]] = list(data=phase.gesture, class.no=phase.gesture.class, formula=phase.gesture.formula)
datasets[["Digits"]] = list(data=digits, class.no=digits.class, formula=digits.formula)
datasets[["EegEye"]] = list(data=eeg.eye, class.no=eeg.eye.class, formula=eeg.eye.formula)
datasets[["Htru"]] = list(data=htru, class.no=htru.class, formula=htru.formula)
datasets[["Telescope"]] = list(data=telescope, class.no=telescope.class, formula=telescope.formula)
datasets[["LetterRecognition"]] = list(data=letter.recognition, class.no=letter.recognition.class, formula=letter.recognition.formula)
datasets[["Statlog"]] = list(data=statlog, class.no=statlog.class, formula=statlog.formula)
datasets[["SensorlessDrive"]] = list(data=sensorless.drive, class.no=sensorless.drive.class, formula=sensorless.drive.formula)
datasets[["Skin"]] = list(data=skin, class.no=skin.class, formula=skin.formula)
datasets[["DiabetesPic"]] = list(data=diabetes.pic, class.no=diabetes.pic.class, formula=diabetes.pic.formula)
datasets[["IndianLiver"]] = list(data=indian.liver, class.no=indian.liver.class, formula=indian.liver.formula)
datasets[["DrugConsumption"]] = list(data=drug.consumption, class.no=drug.consumption.class, formula=drug.consumption.formula)
datasets[["Musk"]] = list(data=musk, class.no=musk.class, formula=musk.formula)
datasets[["Image"]] = list(data=image, class.no=image.class, formula=image.formula)
datasets[["Frogs"]] = list(data=frogs, class.no=frogs.class, formula=frogs.formula)
datasets[["Bankotes"]] = list(data=banknotes, class.no=banknotes.class, formula=banknotes.formula)

cross_validation.fun <- function(Dataset, class.no, fun.train, fun.test, fun.calculate.error, k=5, algorithm = "other"){
    require(cvTools)
    folds <- cvFolds(NROW(Dataset), K=k)
    error <- c()
    time.train <- c()
    time.test <-c()
    for (i in 1:k){
        train <- Dataset[folds$subsets[folds$which != i], ]
        validation <- Dataset[folds$subsets[folds$which == i], ] #Set the validation set
        validation.data <- validation[, -class.no]
        validation.test <- validation[, class.no]
        if (algorithm=="REBMIX"){
            require(rebmix)
            data <- rbind(cbind(train, type=1), cbind(validation, type=2))
            Data <- split(p=list(type=ncol(data), train = 1, test= 1),  Dataset = data,  class.no)#Set the training set
            train <- Data@train
            validation.data <- Data@test
            validation.test <- Data@Zt
            #names(train) <- as.integer(names(train))
            }
        t.train <- system.time(model <- fun.train(train))["elapsed"]
        if (algorithm=="REBMIX" | algorithm=="MCLUST"){
            t.test <- system.time(summary <- fun.test(model, validation.data, validation.test))["elapsed"]
            if (algorithm=="REBMIX"){
                error <- c(error, summary@Error)
            }
            else{
                error <- c(error, summary$err.newdata)
            }
        }
        else{
            t.test <- system.time(result <- fun.test(model, validation.data))["elapsed"]
            error <- c(error, fun.calculate.error(result, validation.test))
            
        }
        time.train <- c(time.train, t.train)
        time.test <- c(time.test, t.test)
    }
    return(list(error=error, time.train=time.train, time.test = time.test))
}
repeat.cross.validation <- function(Dataset, class.no, fun.train, fun.test, fun.calculate.error, k=5, no.iterations=10, algorithm = "other"){
    error <- c()
    time.train <- c()
    time.test <-c()
    for (i in 1:no.iterations){
        set.seed(i)
        results <- cross_validation.fun(Dataset, class.no, fun.train, fun.test, fun.calculate.error, k=k, algorithm = algorithm)
        error <- c(error, results$error)
        time.train <- as.double(c(time.train, results$time.train))
        time.test <- as.double(c(time.test, results$time.test))
    }
    return(list(error=error, time.train=time.train, time.test = time.test))
}

lda.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    
    require(MASS)
    fun.train <- function(x) lda(Dataset.formula, x)
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result$class))/length(validation.test)) 
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
}
qda.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    
    require(MASS)
    fun.train <- function(x) qda(Dataset.formula, x)
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result$class))/length(validation.test)) 
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
}
mda.lin.estimation.impr <- function(train.data, class.no, class.name, subclasses = 2:9){
    require(mda)
    error <- 1
    model.fin <- 0
    train.dataclass <- train.data[, -c(class.no)]
    train.class <- factor(train.data[, class.no])
    for (j in subclasses){
        model <- mda(class.name,train.data, subclasses=j)
        p<-predict(model, newdata = train.dataclass, type="class")
        error.method.new <- (1 - length(which(train.class==p))/length(train.class))
        if (error.method.new < error){
            model.fin <- model
            error <- error.method.new
            }
        }
    return(model.fin)
    }
mda.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(mda)
    fun.train <- function(x) mda.lin.estimation.impr(x, Dataset.class.no, Dataset.formula)
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result))/length(validation.test))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
        
}
multinom.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(nnet)
    fun.train <- function(x) multinom(Dataset.formula, x)
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result))/length(validation.test))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
        
}
nnet.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(nnet)
    scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
    Dataset.data[, -Dataset.class.no] <- as.data.frame(lapply(Dataset.data[, -Dataset.class.no], scl))
    fun.train <- function(x) nnet(x[, -Dataset.class.no],class.ind(x[, Dataset.class.no]) ,size = 10,censored=TRUE)
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result))/length(validation.test))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
        
}
svm.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require("e1071")
    fun.train <- function(x) svm(x=x[, -Dataset.class.no],y=x[, Dataset.class.no], type="C-classification")
    fun.test <- function(model, x) predict(model, newdata=x, type="class")
    fun.calculate.error <- function(result, validation.test) (1 - length(which(validation.test==result))/length(validation.test))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test, fun.calculate.error, k = k, no.iterations=no.iterations)
    return(errors)
        
}
rebmix.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(rebmix)
    fun.train <- function (x) REBMIX(model = "REBMIX", Dataset = x,
                    Preprocessing = "histogram", cmax = 9  ,pdf=rep("n",ncol(Dataset.data)-1), Criterion = "BIC")
    fun.test <- function (model, x, y) RCLSMIX(model = "RCLSMIX", x = list(model), Dataset = x, Zt = factor(y))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test,algorithm="REBMIX", k = k, no.iterations=no.iterations)
    return(errors)
        
}
rebmvnorm.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(rebmix)
    fun.train <- function (x) REBMIX(model = "REBMVNORM", Dataset = x,
                    Preprocessing = "histogram", cmax = 9  , Criterion = "BIC")
    fun.test <- function (model, x, y) RCLSMIX(model = "RCLSMVNORM", x = list(model), Dataset = x, Zt = factor(y))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test,algorithm="REBMIX", k = k, no.iterations=no.iterations)
    return(errors)
        
}
mclust.estimates <- function(Dataset.name, Dataset.data, Dataset.class.no, Dataset.formula, k=5, no.iterations=10){
    require(mclust)
    fun.train <- function (x) MclustDA(x[, -Dataset.class.no], x[, Dataset.class.no])
    fun.test <- function (model, x, y) summary(model, newdata = as.matrix(x), newclass = factor(y))
    errors <- repeat.cross.validation(Dataset.data, Dataset.class.no, fun.train, fun.test,algorithm="MCLUST", k = k, no.iterations=no.iterations)
    return(errors)
        
}
