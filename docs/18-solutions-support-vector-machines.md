# Solutions ch. 6 - Support vector machines {#solutions-svm}

Solutions to exercises of chapter \@ref(svm).

## Exercise 1

Load required libraries

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(doMC)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
library(pROC)
```

```
## Type 'citation("pROC")' for a citation.
```

```
## 
## Attaching package: 'pROC'
```

```
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

```r
library(e1071)
```

Define a radial SVM using the e1071 library

```r
svmRadialE1071 <- list(
  label = "Support Vector Machines with Radial Kernel - e1071",
  library = "e1071",
  type = c("Regression", "Classification"),
  parameters = data.frame(parameter="cost",
                          class="numeric",
                          label="Cost"),
  grid = function (x, y, len = NULL, search = "grid") 
    {
      if (search == "grid") {
        out <- expand.grid(cost = 2^((1:len) - 3))
      }
      else {
        out <- data.frame(cost = 2^runif(len, min = -5, max = 10))
      }
      out
    },
  loop=NULL,
  fit=function (x, y, wts, param, lev, last, classProbs, ...) 
    {
      if (any(names(list(...)) == "probability") | is.numeric(y)) {
        out <- e1071::svm(x = as.matrix(x), y = y, kernel = "radial", 
                          cost = param$cost, ...)
      }
      else {
        out <- e1071::svm(x = as.matrix(x), y = y, kernel = "radial", 
                          cost = param$cost, probability = classProbs, ...)
      }
      out
    },
  predict = function (modelFit, newdata, submodels = NULL) 
    {
      predict(modelFit, newdata)
    },
  prob = function (modelFit, newdata, submodels = NULL) 
    {
      out <- predict(modelFit, newdata, probability = TRUE)
      attr(out, "probabilities")
    },
  predictors = function (x, ...) 
    {
      out <- if (!is.null(x$terms)) 
        predictors.terms(x$terms)
      else x$xNames
      if (is.null(out)) 
        out <- names(attr(x, "scaling")$x.scale$`scaled:center`)
      if (is.null(out)) 
        out <- NA
      out
    },
  tags = c("Kernel Methods", "Support Vector Machines", "Regression", "Classifier", "Robust Methods"),
  levels = function(x) x$levels,
  sort = function(x)
  {
    x[order(x$cost), ]
  }
)
```

Setup parallel processing

```r
registerDoMC(detectCores())
getDoParWorkers()
```

```
## [1] 8
```

Load data

```r
data(segmentationData)
```


```r
segClass <- segmentationData$Class
```

Extract predictors from segmentationData

```r
segData <- segmentationData[,4:59]
```

Partition data

```r
set.seed(42)
trainIndex <- createDataPartition(y=segClass, times=1, p=0.5, list=F)
segDataTrain <- segData[trainIndex,]
segDataTest <- segData[-trainIndex,]
segClassTrain <- segClass[trainIndex]
segClassTest <- segClass[-trainIndex]
```

Set seeds for reproducibility (optional). We will be trying 9 values of the tuning parameter with 5 repeats of 10 fold cross-validation, so we need the following list of seeds.

```r
set.seed(42)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 9)
seeds[[51]] <- sample.int(1000,1)
```

We will pass the twoClassSummary function into model training through **trainControl**. Additionally we would like the model to predict class probabilities so that we can calculate the ROC curve, so we use the **classProbs** option. 

```r
cvCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 5,
                       number = 10,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE,
                       seeds=seeds)
```

Tune SVM over the cost parameter. The default grid of cost parameters start at 0.25 and double at each iteration. Choosing ```tuneLength = 9``` will give us cost parameters of 0.25, 0.5, 1, 2, 4, 8, 16, 32 and 64. The train function will calculate an appropriate value of sigma (the kernel parameter) from the data.

```r
svmTune <- train(x = segDataTrain,
                 y = segClassTrain,
                 method = svmRadialE1071,
                 tuneLength = 9,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = cvCtrl)

svmTune
```

```
## Support Vector Machines with Radial Kernel - e1071 
## 
## 1010 samples
##   56 predictor
##    2 classes: 'PS', 'WS' 
## 
## Pre-processing: centered (56), scaled (56) 
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## Summary of sample sizes: 909, 909, 909, 909, 909, 909, ... 
## Resampling results across tuning parameters:
## 
##   cost   ROC        Sens       Spec     
##    0.25  0.8769145  0.8772308  0.6605556
##    0.50  0.8814957  0.8827692  0.6961111
##    1.00  0.8818205  0.8735385  0.7266667
##    2.00  0.8796581  0.8655385  0.7350000
##    4.00  0.8706410  0.8569231  0.7272222
##    8.00  0.8564786  0.8483077  0.7055556
##   16.00  0.8457009  0.8455385  0.7061111
##   32.00  0.8333077  0.8323077  0.6961111
##   64.00  0.8246923  0.8243077  0.6844444
## 
## ROC was used to select the optimal model using the largest value.
## The final value used for the model was cost = 1.
```


```r
svmTune$finalModel
```

```
## 
## Call:
## svm.default(x = as.matrix(x), y = y, kernel = "radial", cost = param$cost, 
##     probability = classProbs)
## 
## 
## Parameters:
##    SVM-Type:  C-classification 
##  SVM-Kernel:  radial 
##        cost:  1 
## 
## Number of Support Vectors:  540
```

SVM accuracy profile

```r
plot(svmTune, metric = "ROC", scales = list(x = list(log =2)))
```

<div class="figure" style="text-align: center">
<img src="18-solutions-support-vector-machines_files/figure-html/svmAccuracyProfileCellSegment-1.png" alt="SVM accuracy profile." width="80%" />
<p class="caption">SVM accuracy profile.</p>
</div>

Test set results

```r
#segDataTest <- predict(transformations, segDataTest)
svmPred <- predict(svmTune, segDataTest)
confusionMatrix(svmPred, segClassTest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  PS  WS
##         PS 569 101
##         WS  81 258
##                                           
##                Accuracy : 0.8196          
##                  95% CI : (0.7945, 0.8429)
##     No Information Rate : 0.6442          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.6015          
##                                           
##  Mcnemar's Test P-Value : 0.159           
##                                           
##             Sensitivity : 0.8754          
##             Specificity : 0.7187          
##          Pos Pred Value : 0.8493          
##          Neg Pred Value : 0.7611          
##              Prevalence : 0.6442          
##          Detection Rate : 0.5639          
##    Detection Prevalence : 0.6640          
##       Balanced Accuracy : 0.7970          
##                                           
##        'Positive' Class : PS              
## 
```

Get predicted class probabilities

```r
svmProbs <- predict(svmTune, segDataTest, type="prob")
head(svmProbs)
```

```
##           PS         WS
## 1  0.8919246 0.10807539
## 5  0.9110754 0.08892455
## 8  0.9678410 0.03215904
## 9  0.5512136 0.44878636
## 10 0.8198217 0.18017825
## 12 0.7684369 0.23156308
```

Build a ROC curve

```r
svmROC <- roc(segClassTest, svmProbs[,"PS"])
```

```
## Setting levels: control = PS, case = WS
```

```
## Setting direction: controls > cases
```

```r
auc(svmROC)
```

```
## Area under the curve: 0.8908
```

Plot ROC curve.

```r
plot(svmROC, type = "S")
```

<div class="figure" style="text-align: center">
<img src="18-solutions-support-vector-machines_files/figure-html/svmROCcurveCellSegment-1.png" alt="SVM ROC curve for cell segmentation data set." width="80%" />
<p class="caption">SVM ROC curve for cell segmentation data set.</p>
</div>

Calculate area under ROC curve

```r
auc(svmROC)
```

```
## Area under the curve: 0.8908
```


