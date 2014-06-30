# This code creates a number of confusion plots as a function of two diffrent independent variables, trainingfn
# and regularization.
# Coded by Max Lotstein

setwd("C:/Users/Max/Documents/GitHub/ml-freiburg/task1/data/")
results <- read.csv("errorTable18-Jun-2014.csv")

# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
rownames <- c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")

# For every trainingfn
for (tf in c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")) {
  # consider the subset of results for this trainingfn
  s <- results[which(results$TrainingFn== tf),] 
  # look at only the last few columns
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  # build a 3 x 3 table (confusion plot)
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  # label the axes of the table
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  # set the output based on the tf
  pdf(file=paste("conf", tf, ".pdf",sep=""),width=8.97,height=5.76)
  # generate a heatmap using the color palette
  heatmap(as.matrix(df), Rowv=NA, main = tf, Colv=NA, col = my_palette, scale="none", margins=c(5,10))
  dev.off()
}

regularizers <- c(0.0, 0.1, 0.2, 0.5, 1.0)

# for every regularization value
for (r in regularizers) {
  # get the subset of data that use this value
  s <- results[which(results$Regularization == r),] 
  # consider only the last few columns
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  # build a 3 x 3 table (confusion plot)
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  # set the names of the axes
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  pdf(file=paste("conf", gsub("/.", "_", r), ".pdf",sep=""),width=8.97,height=5.76)
  heatmap(as.matrix(df), Rowv=NA, main = r, Colv=NA, col = my_palette, scale="none", margins=c(5,10))
  dev.off()
}
