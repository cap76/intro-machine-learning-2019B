library(ggplot2)
library(GGally)
if(!require(plyr)){install.packages("plyr")}
library(plyr)
if(!require(dplyr)){install.packages("dplyr")}
library(dplyr)
if(!require(reshape2)){install.packages("reshape2")}
library(reshape2)
if(!require(caret)){install.packages("caret")}
library(caret)
if(!require(corrplot)){install.packages("corrplot")}
library(corrplot)

#loading data and removing index column
setwd("C:/Users/ecw63/ml_teaching/")
covid.data <- read.csv("covid_final.csv",header=T)
covid.data = covid.data[,2:ncol(covid.data)]

#summary stats
head(covid.data)
summary(covid.data)
str(covid.data)
covid.data$result = as.factor(covid.data$result)

#violin plots for location, country, symptom1 and symptom2 against result
ggplot(covid.data, aes(result, location, fill=result)) +
  geom_violin(aes(color = result), trim = T)+
  scale_y_continuous("Location", breaks= seq(0,150, by=10))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")

ggplot(covid.data, aes(result, country, fill=result)) +
  geom_violin(aes(color = result), trim = T)+
  scale_y_continuous("Country", breaks= seq(0,50, by=5))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")

ggplot(covid.data, aes(result, symptom1, fill=result)) +
  geom_violin(aes(color = result), trim = T)+
  scale_y_continuous("Symp1", breaks= seq(0,25, by=2))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")

ggplot(covid.data, aes(result, symptom2, fill=result)) +
  geom_violin(aes(color = result), trim = T)+
  scale_y_continuous("Symp2", breaks= seq(0,40, by=2))+
  geom_boxplot(width=0.1)+
  theme(legend.position="none")

#jitter plot showing distribution of each variable, coloured by result
pdf("vio_jitter_covid.pdf", width = 10, height = 5)
exploratory.covid <- melt(covid.data)
exploratory.covid %>%
  ggplot(aes(x = factor(variable), y = value)) +
  geom_violin() +
  geom_jitter(height = 0, width = 0.1, aes(colour = result), alpha = 0.7) +
  theme_minimal()
dev.off()

#ggpairs plot summary
pdf("all_info_covid.pdf", width=20, height=20)
ggpairs(covid.data, ggplot2::aes(colour = result, alpha = 0.4))
dev.off()

covidClass <- covid.data$result
covidData <- covid.data[,1:13]

#splitting into training and test
set.seed(42)
trainIndex <- createDataPartition(y=covidClass, times=1, p=0.7, list=F)
classTrain <- covidClass[trainIndex]
dataTrain <- covidData[trainIndex,]
classTest <- covidClass[-trainIndex]
dataTest <- covidData[-trainIndex,]

#identifying zero variance or near zero variance variables
nzv <- nearZeroVar(dataTrain, saveMetrics=T)
print(nzv)
print(rownames(nzv[nzv$nzv==TRUE,]))
listtoexclude = c(rownames(nzv[nzv$nzv==TRUE ,]))
listtoexclude = c(listtoexclude,rownames(nzv[nzv$zeroVar==TRUE ,]))


summary(dataTrain)

#boxplots showing distribution of each variable in the training set
featurePlot(x = dataTrain,
            y = classTrain,
            plot = "box",
            ## Pass in options to bwplot()
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3,3))

#density plot for each variable in training set
featurePlot(x = dataTrain,
            y = classTrain,
            plot = "density",
            ## Pass in options to xyplot() to
            ## make it prettier
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            adjust = 1.5,
            pch = "|",
            layout = c(3, 3),
            auto.key = list(columns = 3))


#correlation plot comparing each pair of variables
corMat <- cor(dataTrain)
corrplot(corMat, order="hclust", tl.cex=1)

#identifying highly correlated variables
highCorr <- findCorrelation(corMat, cutoff=0.5)
length(highCorr)
names(dataTrain)[highCorr]
listtoexclude = c(listtoexclude,names(dataTrain)[highCorr])

tuneParam <- data.frame(k=seq(1,50,2))

#setting seeds
set.seed(42)
seeds <- vector(mode = "list", length = 101)
for(i in 1:100) seeds[[i]] <- sample.int(1000, length(tuneParam$k))
seeds[[101]] <- sample.int(1000,1)

train_ctrl <- trainControl(method="repeatedcv",
                           number = 10,
                           repeats = 10,
                           preProcOptions=list(cutoff=0.75),
                           seeds = seeds)

#running kNN for multiple k values to choose the appropriate one
knnFit <- train(dataTrain, classTrain,
                method="knn",
                preProcess = c("center", "scale", "corr"),
                tuneGrid=tuneParam,
                trControl=train_ctrl)
knnFit
#plotting accuracy against k
plot(knnFit)
plot(knnFit,metric="Kappa")

#testing the resulting model on the test set
test_pred <- predict(knnFit, dataTest)
confusionMatrix(test_pred, classTest)

#removing zero var, near zero var and highly correlated variables
dataTrain_restricted = dataTrain[, -which(names(dataTrain) %in% listtoexclude)]
dataTest_restricted = dataTest[, -which(names(dataTest) %in% listtoexclude)]
train_ctrl_restricted <- trainControl(method="repeatedcv",
                           number = 10,
                           repeats = 10,
                           preProcOptions=list(cutoff=0.75),
                           seeds = seeds)

#running kNN for multiple k values to choose the appropriate one
knnFit_restricted <- train(dataTrain_restricted, classTrain,
                method="knn",
                preProcess = c("center", "scale", "corr"),
                tuneGrid=tuneParam,
                trControl=train_ctrl_restricted)
plot(knnFit_restricted)
plot(knnFit_restricted,metric="Kappa")
