Machine Learning Project
================
AAleid
December 7, 2018

### Background and Objective

#### Activity trackers are commonly used now.They measure body movements at certain patterns to detect certain types of activities. In this report, the quality of these types of activities will be studied and predicted, using data obtained from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

### Loading Libraries

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(rpart)
library(rattle)
```

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(gbm)
```

    ## Loaded gbm 2.1.4

### Importing Data

``` r
train<-read.csv(url ("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))

test<-read.csv(url("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

#cleaning data
train<- train[, colSums(is.na(train)) == 0]
Remove <- which(colSums(is.na(train)|train=="")>0.9*dim(train)[1]) 
train <- train[,-Remove]


train<- train[, -c(1:7)]
              
              

Remove1 <- which(colSums(is.na(test) |test=="")>0.9*dim(test)[1]) 
test <- test[,-Remove1]


test<- test[, -1]

dim(train)
```

    ## [1] 19622    53

``` r
dim(test)
```

    ## [1] 20 59

``` r
set.seed(1234)
training <- createDataPartition(train$classe, p=0.6, list=FALSE)
trainingset <- train[training, ]
testingset <- train[-training, ]
dim(trainingset)
```

    ## [1] 11776    53

``` r
dim(testingset)
```

    ## [1] 7846   53

#### To enhance the performance of our model, we will use 10 folds cross validation

#### We will use:

-   classification tree
-   random forest
-   gradient boosting method

### 1.Classification Tree

``` r
cv<-trainControl(method="cv", number=10)
CT <- train(classe~., data=trainingset, method="rpart")
fancyRpartPlot(CT$finalModel)
```

![](machineLearning_files/figure-markdown_github/unnamed-chunk-3-1.png)

#### Validate our model with the test set

``` r
predic <- predict(CT,newdata=testingset)

conf <- confusionMatrix(testingset$classe,predic)

conf
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2029   44  155    0    4
    ##          B  638  505  375    0    0
    ##          C  644   49  675    0    0
    ##          D  567  232  487    0    0
    ##          E  209  211  383    0  639
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4904          
    ##                  95% CI : (0.4793, 0.5016)
    ##     No Information Rate : 0.5209          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.3339          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4965  0.48511  0.32530       NA  0.99378
    ## Specificity            0.9460  0.85114  0.87992   0.8361  0.88852
    ## Pos Pred Value         0.9091  0.33267  0.49342       NA  0.44313
    ## Neg Pred Value         0.6334  0.91530  0.78388       NA  0.99938
    ## Prevalence             0.5209  0.13268  0.26447   0.0000  0.08195
    ## Detection Rate         0.2586  0.06436  0.08603   0.0000  0.08144
    ## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
    ## Balanced Accuracy      0.7212  0.66812  0.60261       NA  0.94115

``` r
conf$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2029   44  155    0    4
    ##          B  638  505  375    0    0
    ##          C  644   49  675    0    0
    ##          D  567  232  487    0    0
    ##          E  209  211  383    0  639

``` r
conf$overall[1]
```

    ## Accuracy 
    ## 0.490441

#### the accuracy is 49% (low)

------------------------------------------------------------------------

### 2. Random Forest

``` r
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)

modRF1 <- train(classe ~ ., data=trainingset, method="rf", trControl=controlRF)

modRF1$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 0.86%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3344    2    0    0    2 0.001194743
    ## B   19 2254    6    0    0 0.010969724
    ## C    0   21 2029    4    0 0.012171373
    ## D    0    0   36 1893    1 0.019170984
    ## E    0    0    3    7 2155 0.004618938

#### Validate our model with the test set

``` r
predictRF <- predict(modRF1, newdata=testingset)
cmRF <- confusionMatrix(predictRF, testingset$classe)
cmRF
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2232   12    0    0    0
    ##          B    0 1504   10    0    0
    ##          C    0    2 1353   29    3
    ##          D    0    0    5 1255    3
    ##          E    0    0    0    2 1436
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9916          
    ##                  95% CI : (0.9893, 0.9935)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9894          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9908   0.9890   0.9759   0.9958
    ## Specificity            0.9979   0.9984   0.9948   0.9988   0.9997
    ## Pos Pred Value         0.9947   0.9934   0.9755   0.9937   0.9986
    ## Neg Pred Value         1.0000   0.9978   0.9977   0.9953   0.9991
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1917   0.1724   0.1600   0.1830
    ## Detection Prevalence   0.2860   0.1930   0.1768   0.1610   0.1833
    ## Balanced Accuracy      0.9989   0.9946   0.9919   0.9873   0.9978

#### Accuracy is 99.1%

``` r
plot(modRF1)
```

![](machineLearning_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
plot(cmRF$table, col = cmRF$byClass, main = paste("Random Forest Accuracy ", round(cmRF$overall['Accuracy'], 4)))
```

![](machineLearning_files/figure-markdown_github/unnamed-chunk-7-2.png)

### 3. gradient boosting method

``` r
set.seed(12345)
controlG <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modG  <- train(classe ~ ., data=trainingset, method = "gbm", trControl = controlG, verbose = FALSE)
modG$finalModel
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 52 predictors of which 40 had non-zero influence.

``` r
print(modG)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 11776 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 1 times) 
    ## Summary of sample sizes: 9421, 9421, 9422, 9420, 9420 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7461774  0.6783375
    ##   1                  100      0.8176785  0.7692818
    ##   1                  150      0.8505445  0.8109745
    ##   2                   50      0.8533444  0.8141926
    ##   2                  100      0.9043821  0.8790101
    ##   2                  150      0.9324893  0.9145780
    ##   3                   50      0.8952948  0.8674332
    ##   3                  100      0.9395377  0.9234972
    ##   3                  150      0.9598339  0.9491847
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 150,
    ##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

#### Validate our model with the test set

``` r
predictG <- predict(modG, newdata=testingset)
cmG <- confusionMatrix(predictG, testingset$classe)
cmG
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2208   53    0    0    1
    ##          B   16 1424   51    6   19
    ##          C    2   40 1292   41    7
    ##          D    4    0   21 1233   12
    ##          E    2    1    4    6 1403
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9635          
    ##                  95% CI : (0.9592, 0.9676)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9539          
    ##  Mcnemar's Test P-Value : 5.666e-09       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9892   0.9381   0.9444   0.9588   0.9730
    ## Specificity            0.9904   0.9855   0.9861   0.9944   0.9980
    ## Pos Pred Value         0.9761   0.9393   0.9349   0.9709   0.9908
    ## Neg Pred Value         0.9957   0.9852   0.9882   0.9919   0.9939
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2814   0.1815   0.1647   0.1572   0.1788
    ## Detection Prevalence   0.2883   0.1932   0.1761   0.1619   0.1805
    ## Balanced Accuracy      0.9898   0.9618   0.9653   0.9766   0.9855

#### Accuracy is 96.2%

------------------------------------------------------------------------

\*the highest accuracy is RANDOM FOREST, so it will be used to validate the data

``` r
final<-predict(modRF1, newdata = test)
final
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
