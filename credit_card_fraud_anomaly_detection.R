
install.packages("caret")
install.packages("ROSE")
install.packages("ROCR")

# ------------------------------------------------------------------------------
# Load data
df <- read.csv("Data/creditcard.csv")

# Explore data
str(df)

# ------------------------------------------------------------------------------
# Exploratory Visualisations
ggplot(fraud, aes(x = Time, y = Amount)) +
  geom_point() +
  ggtitle("Fraudulent Transactions")

# ------------------------------------------------------------------------------
# Correlation Heatmap
library(GGally)

corr <- ggcorr(credit_card_data, labels = T)
corr

# Extract correlation coefficients of 'Class'
rank <- as.data.frame(cor(credit_card_data, credit_card_data$Class))
rank <- as.data.frame(rank[order(-rank$V1),])

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

svm_train <- svm(Class ~ ., train, type = "C-classification", kernel = "linear") # Training with entire dataset
table(train$Class) # Confusion Matrix

# ------------------------------------------------------------------------------
# Predictive Model Evaluation with Test Data
library(caret)

confusionMatrix(predict(svm_train, test), test$Class, positive = '1') # Testing with entire dataset

# ------------------------------------------------------------------------------
# Under-Sampling
library(ROSE)

under <- ovun.sample(Class ~ ., train, method = "under", N = 688)$data
table(under$Class) # Confusion Matrix

svm_under <- svm(Class ~ ., under, type = "C-classification", kernel = "linear") # Training with under-sampled data

confusionMatrix(predict(svm_under, test), test$Class, positive = '1') # Testing with under-sampled data

# ------------------------------------------------------------------------------
# Over-Sampling
#over <- ovun.sample(Class ~ ., data = train, method = "over", N = 397792)$data
#table(over$Class)

#svm_over <- svm(Class ~ ., over, type = "C-classification", kernel = "linear")

# ------------------------------------------------------------------------------
# Both
#both <- ovun.sample(Class ~ ., train, method = "both", p = 0.5, seed = 222, N = 199240)$data
#table(both$Class)

#svm_both <- svm(Class ~ ., both, type = "C-classification", kernel = "linear")

# ------------------------------------------------------------------------------
# ROC Curve


