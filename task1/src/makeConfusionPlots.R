setwd("~/GitHub/ml-freiburg/task1/src/Run2")
results <- read.csv("errorTable18-Jun-2014.csv")

# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
rownames <- c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")

for (tf in c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")) {
  s <- subset(results, TrainingFn=tf)
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  heatmap(as.matrix(df), Rowv=NA, main = tf, Colv=NA, col = my_palette, scale="column", margins=c(5,10))
}

regularizers <- c(0.0, 0.1, 0.2, 0.5, 1.0)

for (r in regularizers) {
  s <- subset(results, Regularization=r)
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  heatmap(as.matrix(df), Rowv=NA, main = r, Colv=NA, col = my_palette, scale="column", margins=c(5,10))
}