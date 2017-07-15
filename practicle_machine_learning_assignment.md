practicle machine learning assinment
================

<table style="width:33%;">
<colgroup>
<col width="33%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="left">Background Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement Â– a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.</td>
</tr>
<tr class="even">
<td align="left">One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this data set, the participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a> (see the section on the Weight Lifting Exercise Dataset).</td>
</tr>
<tr class="odd">
<td align="left">In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants toto predict the manner in which praticipants did the exercise.</td>
</tr>
<tr class="even">
<td align="left">The dependent variable or response is the “classe” variable in the training set.</td>
</tr>
</tbody>
</table>

Data
----

Download and load the data.

------------------------------------------------------------------------

``` r
trainingOrg = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testingOrg = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(trainingOrg)
```

    ## [1] 19622   160

Pre-screening the data
----------------------

<table style="width:35%;">
<colgroup>
<col width="34%" />
</colgroup>
<tbody>
<tr class="odd">
<td>There are several approaches for reducing the number of predictors.</td>
</tr>
<tr class="even">
<td>Remove variables that we believe have too many NA values.</td>
</tr>
</tbody>
</table>

``` r
training.dena <- trainingOrg[ , colSums(is.na(trainingOrg)) == 0]
dim(training.dena)
```

    ## [1] 19622    60

Remove unrelevant variables There are some unrelevant variables that can be removed as they are unlikely to be related to dependent variable.
---------------------------------------------------------------------------------------------------------------------------------------------

``` r
remove = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
training.dere <- training.dena[, -which(names(training.dena) %in% remove)]
dim(training.dere)
```

    ## [1] 19622    53

Check the variables that have extremely low variance (this method is useful nearZeroVar() )
-------------------------------------------------------------------------------------------

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.3.2

``` r
## Loading required package: lattice
## Loading required package: ggplot2
# only numeric variabls can be evaluated in this way.

zeroVar= nearZeroVar(training.dere[sapply(training.dere, is.numeric)], saveMetrics = TRUE)
training.nonzerovar = training.dere[,zeroVar[, 'nzv']==0]
dim(training.nonzerovar)
```

    ## [1] 19622    53

Remove highly correlated variables 90% (using for example findCorrelation() )
-----------------------------------------------------------------------------

``` r
corrMatrix <- cor(na.omit(training.nonzerovar[sapply(training.nonzerovar, is.numeric)]))
dim(corrMatrix)
```

    ## [1] 52 52

``` r
# there are 52 variables.
corrDF <- expand.grid(row = 1:52, col = 1:52)
corrDF$correlation <- as.vector(corrMatrix)
levelplot(correlation ~ row+ col, corrDF)
```

![](practicle_machine_learning_assignment_files/figure-markdown_github/unnamed-chunk-6-1.png)

We are going to remove those variable which have high correlation.
------------------------------------------------------------------

``` r
removecor = findCorrelation(corrMatrix, cutoff = .90, verbose = TRUE)
```

    ## Compare row 10  and column  1 with corr  0.992 
    ##   Means:  0.27 vs 0.168 so flagging column 10 
    ## Compare row 1  and column  9 with corr  0.925 
    ##   Means:  0.25 vs 0.164 so flagging column 1 
    ## Compare row 9  and column  4 with corr  0.928 
    ##   Means:  0.233 vs 0.161 so flagging column 9 
    ## Compare row 8  and column  2 with corr  0.966 
    ##   Means:  0.245 vs 0.157 so flagging column 8 
    ## Compare row 19  and column  18 with corr  0.918 
    ##   Means:  0.091 vs 0.158 so flagging column 18 
    ## Compare row 46  and column  31 with corr  0.914 
    ##   Means:  0.101 vs 0.161 so flagging column 31 
    ## Compare row 46  and column  33 with corr  0.933 
    ##   Means:  0.083 vs 0.164 so flagging column 33 
    ## All correlations <= 0.9

``` r
training.decor = training.nonzerovar[,-removecor]
dim(training.decor)
```

    ## [1] 19622    46

Split data to training and testing for cross validation.
--------------------------------------------------------

``` r
inTrain <- createDataPartition(y=training.decor$classe, p=0.7, list=FALSE)
training <- training.decor[inTrain,]; testing <- training.decor[-inTrain,]
dim(training);dim(testing)
```

    ## [1] 13737    46

    ## [1] 5885   46

Analysis
--------

<table style="width:15%;">
<colgroup>
<col width="15%" />
</colgroup>
<tbody>
<tr class="odd">
<td>Regression Tree</td>
</tr>
<tr class="even">
<td>Now we fit a tree to these data, and summarize and plot it. First, we use the 'tree' package. It is much faster than 'caret' package.</td>
</tr>
</tbody>
</table>

``` r
library(tree)
```

    ## Warning: package 'tree' was built under R version 3.3.3

``` r
set.seed(12345)
tree.training=tree(classe~.,data=training)
summary(tree.training)
```

    ## 
    ## Classification tree:
    ## tree(formula = classe ~ ., data = training)
    ## Variables actually used in tree construction:
    ##  [1] "pitch_forearm"     "magnet_belt_y"     "accel_forearm_z"  
    ##  [4] "magnet_dumbbell_y" "roll_forearm"      "magnet_dumbbell_z"
    ##  [7] "accel_dumbbell_y"  "yaw_belt"          "pitch_belt"       
    ## [10] "accel_forearm_x"   "yaw_dumbbell"      "magnet_arm_x"     
    ## [13] "accel_dumbbell_z"  "gyros_belt_z"     
    ## Number of terminal nodes:  20 
    ## Residual mean deviance:  1.685 = 23110 / 13720 
    ## Misclassification error rate: 0.3378 = 4641 / 13737

``` r
plot(tree.training)
text(tree.training,pretty=0, cex =.8)
```

![](practicle_machine_learning_assignment_files/figure-markdown_github/unnamed-chunk-10-1.png)

lets prune it.
--------------

Rpart form Caret
----------------

``` r
library(caret)
modFit <- train(classe ~ .,method="rpart",data=training)
```

    ## Loading required package: rpart

``` r
print(modFit$finalModel)
```

    ## n= 13737 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) pitch_forearm< -26.75 1241   56 A (0.95 0.045 0 0 0) *
    ##    3) pitch_forearm>=-26.75 12496 9775 A (0.22 0.21 0.19 0.18 0.2)  
    ##      6) magnet_belt_y>=555.5 11470 8749 A (0.24 0.23 0.21 0.18 0.15)  
    ##       12) magnet_dumbbell_y< 437.5 9526 6870 A (0.28 0.18 0.24 0.18 0.13)  
    ##         24) roll_forearm< 122.5 6001 3609 A (0.4 0.18 0.18 0.15 0.091) *
    ##         25) roll_forearm>=122.5 3525 2316 C (0.075 0.17 0.34 0.22 0.19)  
    ##           50) accel_forearm_x>=-108.5 2491 1522 C (0.084 0.21 0.39 0.09 0.23) *
    ##           51) accel_forearm_x< -108.5 1034  473 D (0.053 0.086 0.23 0.54 0.086) *
    ##       13) magnet_dumbbell_y>=437.5 1944 1017 B (0.033 0.48 0.041 0.2 0.25) *
    ##      7) magnet_belt_y< 555.5 1026  189 E (0 0.0029 0.0019 0.18 0.82) *

Cross Validation

We are going to check the performance of the tree on the testing data by cross validation.
==========================================================================================

``` r
tree.pred=predict(tree.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate
```

    ## [1] 0.6632116

from caret package
------------------

``` r
tree.pred=predict(modFit,testing)
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate
```

    ## [1] 0.4890399

Pruning tree

This tree was grown to full depth, and might be too variable. We now use Cross Validation to prune it.
------------------------------------------------------------------------------------------------------

``` r
cv.training=cv.tree(tree.training,FUN=prune.misclass)
cv.training
```

    ## $size
    ##  [1] 20 19 18 17 16 15 14 13 10  9  7  5  1
    ## 
    ## $dev
    ##  [1] 4425 4425 4496 4657 5030 5233 5233 5402 6440 6574 7204 7204 9831
    ## 
    ## $k
    ##  [1]     -Inf  25.0000  63.0000 114.0000 133.0000 141.0000 145.0000
    ##  [8] 159.0000 194.3333 205.0000 237.0000 255.5000 659.2500
    ## 
    ## $method
    ## [1] "misclass"
    ## 
    ## attr(,"class")
    ## [1] "prune"         "tree.sequence"

``` r
plot(cv.training)
```

![](practicle_machine_learning_assignment_files/figure-markdown_github/unnamed-chunk-15-1.png)

It shows that when the size of the tree goes down, the deviance goes up. It means the 21 is a good size (i.e. number of terminal nodes) for this tree. We do not need to prune it.

Suppose we prune it at size of nodes at 18.
===========================================

``` r
prune.training=prune.misclass(tree.training,best=18)
```

Now lets evaluate this pruned tree on the test data.
====================================================

``` r
tree.pred=predict(prune.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate
```

    ## [1] 0.6553951

0.66 is a little less than 0.70, so pruning did not hurt us with repect to misclassification errors, and gave us a simpler tree. We use less predictors to get almost the same result. By pruning, we got a shallower tree, which is easier to interpret.

The single tree is not good enough, so we are going to use bootstrap to improve the accuracy. We are going to try random forests.
=================================================================================================================================

Random Forests
==============

=============== Random forests build lots of bushy trees, and then average them to reduce the variance. =======================================================================================

``` r
require(randomForest)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
set.seed(12345)
```

Lets fit a random forest and see how well it performs.
======================================================

``` r
rf.training=randomForest(classe~.,data=training,ntree=100, importance=TRUE)
rf.training
```

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, ntree = 100,      importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 100
    ## No. of variables tried at each split: 6
    ## 
    ##         OOB estimate of  error rate: 0.72%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 3900    3    1    0    2 0.001536098
    ## B   21 2625   12    0    0 0.012415350
    ## C    0   14 2381    0    1 0.006260434
    ## D    0    0   30 2218    4 0.015097691
    ## E    0    0    3    8 2514 0.004356436

``` r
varImpPlot(rf.training,)
```

![](practicle_machine_learning_assignment_files/figure-markdown_github/unnamed-chunk-19-1.png)

we can see which variables have higher impact on the prediction.

Out-of Sample Accuracy
======================

Our Random Forest model shows OOB estimate of error rate: 0.72% for the training data. Now we will predict it for out-of sample accuracy.

Now lets evaluate this tree on the test data.
=============================================

``` r
tree.pred=predict(rf.training,testing,type="class")
predMatrix = with(testing,table(tree.pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix)) # error rate
```

    ## [1] 0.9949023

0.99 means we got a very accurate estimate.

No. of variables tried at each split: 6. It means every time we only randomly use 6 predictors to grow the tree. Since p = 43, we can have it from 1 to 43, but it seems 6 is enough to get the good result.
============================================================================================================================================================================================================

Conclusion
==========

=========== Now we can predict the testing data from the website. ======================================================

``` r
answers <- predict(rf.training, testingOrg)
answers
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
