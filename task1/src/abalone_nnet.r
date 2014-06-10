library(nnet)
setwd("~/uni-freiburg/ss2014/machineLearning/ml-freiburg/task1/")
data <- read.csv("data//abalone.data")
names(data) <- c("sex", "length", "diameter", "height", "w_weight", "v_weight",
                 "sh_weight", "s_weight", "rings")

# data preprocessing, rescaling, etc.
data[,1] <- as.character(data[,1])
data[data$sex == "M",1] <- 1
data[data$sex == "F",1] <- 2
data[data$sex == "I",1] <- 0
data[,1] <- as.numeric(data[,1])

data$age1 <- ifelse(data$rings <= 8, 1, 0)
data$age2 <- ifelse(data$rings == 9 | data$rings == 10, 1, 0)
data$age3 <- ifelse(data$rings >= 11, 1, 0)
data <- subset(data, select = -rings)

train <- data[1:3133,]
test <- data[3134:nrow(data),]
for(i in 1:(ncol(train) - 3)) {
  range <- max(train[,i]) - min(train[,i])
  min <- min(train[,i])
  train[,i] <- (train[,i] - min) / range
}

# train the neural network
inputs <- train[,1:8]
targets <- train[,9:11]
n <- nnet(inputs, targets, size=20, maxit=2000)

# predict response variable for the test set
test_inputs <- test[,1:8]
test_targets <- test[,9:11]
for(i in 1:ncol(test_inputs)) {
  range <- max(test_inputs[,i]) - min(test_inputs[,i])
  min <- min(test_inputs[,i])
  test_inputs[,i] <- (test_inputs[,i] - min) / range
}
results <- round(predict(n, test_inputs))
no_correct <- sum(results[,1] == 1 & test_targets[,1] == results[,1])
no_correct <- no_correct + sum(results[,2] == 1 & test_targets[,2] == results[,2])
no_correct <- no_correct + sum(results[,3] == 1 & test_targets[,3] == results[,3])

error <- abs(1 - no_correct / nrow(test_targets))

