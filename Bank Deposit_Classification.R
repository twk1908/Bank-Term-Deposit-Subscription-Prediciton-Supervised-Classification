# Project: Prediction of Bank-Term Deposit Subscription (Supervised-Classification)
# Full Name: Tan Wai Kit

#R-code for bank marketing's dataset
dfbank1=read.table("bank.csv",sep=";",header = T, stringsAsFactors = T)
dfbank2=read.table("bank.csv",sep=";",header = T)

summary(dfbank1)
## There are negative values in "pdays"
str(dfbank1)
sum(is.na(dfbank1))
dfbank3 <- dfbank1
summary(dfbank3)

# Preparation of Nominal Data
dfbank3$job <- as.numeric(as.factor(dfbank3$job))
dfbank3$marital <- as.numeric(as.factor(dfbank3$marital))
dfbank3$education <- as.numeric(as.factor(dfbank3$education))
dfbank3$default <- ifelse(dfbank3$default  == "yes", 1, 0)
dfbank3$housing <- ifelse(dfbank3$housing  == "yes", 1, 0)
dfbank3$loan <- ifelse(dfbank3$loan  == "yes", 1, 0)
dfbank3$contact <- as.numeric(as.factor(dfbank3$contact))
dfbank3$month <- as.numeric(as.factor(dfbank3$month))
dfbank3$poutcome <- as.numeric(as.factor(dfbank3$poutcome))
dfbank3$y <- ifelse(dfbank3$y  == "yes", 1 , 0)

# Remedy of negative values in pdays
dfbank3$pdays <- abs(dfbank3$pdays)
dfbank3

# Plotting Correlation
library(GGally)
ggcorr(dfbank3, name = "Correlation", label = T, label_round = 2,
       low = "#FF0000",
       mid = "#FFFFFF",
       high = "#0083EF")

# Removal of pdays due to multicollinearity
dfbank3 <- subset(dfbank3, select = -c(pdays) )
print(dfbank3)

# Stratified Train-Test Split
library(caTools)
set.seed(1908)
dfbank3.train <- sample.split(Y = dfbank3$y, SplitRatio = 0.7)
dfbank3.trainset <- subset(dfbank3, dfbank3.train == T)
dfbank3.testset <- subset(dfbank3, dfbank3.train == F)

# Preparation for Evaluation (AUC of ROC, F1 score)
library(cvms)
library(caret)
library(ggplot2)
library(pROC)

# Logistic Regression
set.seed(1908)
df.log <- glm(y ~ ., family = binomial,  data = dfbank3.trainset)
summary(df.log)
df.log.predict <- predict.glm(df.log, newdata = dfbank3.testset)
df.log.predict2 <- ifelse(df.log.predict > 0.05, "1", "0")
library(gmodels)
CrossTable(x = dfbank3.testset$y, y = df.log.predict2)
logtable <- table(dfbank3.testset$y, df.log.predict2)
prop.table(logtable)

# Logistic Regression Evaluation
df.log.calc <- predict(df.log,dfbank3.testset)
df.log.roc <- roc(predictor = df.log.calc, response = dfbank3.testset$y)
print(df.log.roc)
plot(df.log.roc)
accuracy.log <- mean(df.log.predict2 == dfbank3.testset$y)
print(accuracy.log)

# K Nearest Neighbors k = 3
set.seed(1908)
df.knn.3 <- knn(train = dfbank3.trainset, test = dfbank3.testset, cl = dfbank3.trainset$y, k = 3)
CrossTable(x = dfbank3.testset$y, y = df.knn.3)
knntable3 <- table(dfbank3.testset$y, df.knn.3)
prop.table(knntable3)

# K Nearest Neighbors Evaluation k = 3
df.knn.3.roc <- roc(predictor = ifelse(df.knn.3 == 1, 1, 0), response = dfbank3.testset$y)
print(df.knn.3.roc)
plot(df.knn.3.roc)
accuracy3 <- mean(df.knn.3 == dfbank3.testset$y)
print(accuracy3)

# K Nearest Neighbors k = 4
set.seed(1908)
df.knn.4 <- knn(train = dfbank3.trainset, test = dfbank3.testset, cl = dfbank3.trainset$y, k = sqrt(16))
CrossTable(x = dfbank3.testset$y, y = df.knn.4)
knntable4 <- table(dfbank3.testset$y, df.knn.4)
prop.table(knntable4)

# K Nearest Neighbor Evaluation k = 4
df.knn.4.roc <- roc(predictor = ifelse(df.knn.4 == 1, 1, 0), response = dfbank3.testset$y)
print(df.knn.4.roc)
plot(df.knn.4.roc)
accuracy4 <- mean(df.knn.4 == dfbank3.testset$y)
print(accuracy4)


