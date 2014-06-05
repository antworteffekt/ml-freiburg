
setwd("~/uni-freiburg/ss2014/machineLearning/programming/task1/")
abalone_data <- read.csv("data//abalone.data")
# Sex  	nominal			M, F, and I (infant)
# Length		continuous	mm	Longest shell measurement
# Diameter	continuous	mm	perpendicular to length
# Height		continuous	mm	with meat in shell
# Whole weight	continuous	grams	whole abalone
# Viscera weight	continuous	grams	gut weight (after bleeding)
# Shucked weight	continuous	grams	weight of meat
# Shell weight	continuous	grams	after being dried
# Rings		integer			+1.5 gives the age in years
names(abalone_data) <- c("sex", "length", "diameter", "height", "w_weight", "v_weight",
                 "sh_weight", "s_weight", "rings")
abalone_data$sex <- as.factor(abalone_data$sex)

# pairs(~rings+sex+length+diameter+height+w_weight, data = abalone_data)
# pairs(~height+w_weight+s_weight+v_weight+sh_weight, data = abalone_data)

abalone_data$female <- ifelse(abalone_data$sex == "F", 1, 0)
abalone_data$male <- ifelse(abalone_data$sex == "M", 1, 0)
abalone_data$infant <- ifelse(abalone_data$sex == "I", 1, 0)

abalone_data <- subset(abalone_data, select = -sex)

# split the rings variable into three different binary variables as follows:
# 1-8 first category, 9 and 10 second, +11 third
abalone_data$age1 <- ifelse(abalone_data$rings <= 8, 1, 0)
abalone_data$age2 <- ifelse(abalone_data$rings == 9 || abalone_data == 10, 1, 0)
abalone_data$age3 <- ifelse(abalone_data$rings >= 11, 1, 0)

abalone_data <- subset(abalone_data, select = -rings)

# split train and test sets:
abalone_train <- abalone_data[1:3133,]
abalone_test <- abalone_data[3134:nrow(abalone_data),]

write.table(abalone_train, file = "data/abalone_train.csv", col.names = T, row.names = F)
write.table(abalone_test, file = "data/abalone_test.csv", col.names = T, row.names = F)