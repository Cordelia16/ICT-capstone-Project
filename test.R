
install.packages("caret")
install.packages("ROSE")

# ------------------------------------------------------------------------------
# Load data
df <- read.csv("Data/creditcard.csv")

# Explore data
str(df)

# ------------------------------------------------------------------------------
# Set 'Class' to factor data type
df$Class <- as.factor(df$Class)

# ------------------------------------------------------------------------------
# Data partition
set.seed(123)

ind <- sample(2, nrow(df), replace = TRUE, prob = c(0.7, 0.3)) # 70/30 split
train <- df[ind == 1, ]
test <- df[ind == 2, ]

# ------------------------------------------------------------------------------
# Data for Developing Predictive Model
table(train$Class)
prop.table(table(train$Class))
summary(train)

# ------------------------------------------------------------------------------
# Predictive Model (SVM)
library(e1071)

# svm_train <- svm(Class ~ ., train, type = "C-classification", kernel = "linear")
table(train$Class)

# ------------------------------------------------------------------------------
# Predictive Model Evaluation with Test Data
library(caret)

confusionMatrix(predict(svm_train, test), test$Class, positive = '0')

# ------------------------------------------------------------------------------
# Under-Sampling
library(ROSE)

under <- ovun.sample(Class ~ ., train, method = "under", N = 688)$data
table(under$Class)

svm_under <- svm(Class ~ ., under, type = 'C-classification', kernel = "linear")

confusionMatrix(predict(svm_under, test), test$Class, positive = '1')

# ------------------------------------------------------------------------------
# Over-Sampling
# over <- ovun.sample(Class ~ ., data = train, method = "over", N = 397792)$data
# table(over$Class)

# svm_over <- svm(Class ~ ., over, type = 'C-classification', kernel = "linear")


