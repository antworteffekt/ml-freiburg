setwd("C:/Users/Max/Documents/GitHub/ml-freiburg/task1/data/")
results <- read.csv("errorTable18-Jun-2014.csv")

# creates a own color palette from red to green
my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
rownames <- c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")

for (tf in c("trainbfg","trainbr","traingd","traingda","traingdm","traingdx","trainrp")) {
  s <- results[which(results$TrainingFn== tf),] 
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  pdf(file=paste("conf", tf, ".pdf",sep=""),width=8.97,height=5.76)
  heatmap(as.matrix(df), Rowv=NA, main = tf, Colv=NA, col = my_palette, scale="none", margins=c(5,10))
  dev.off()
}

regularizers <- c(0.0, 0.1, 0.2, 0.5, 1.0)

for (r in regularizers) {
  s <- results[which(results$Regularization == r),] 
  s <- s[c(-1,-2,-3,-4,-5,-6,-7)]
  conf <- as.vector(colMeans(s))
  df <- data.frame(conf[1:3], conf[4:6], conf[7:9])
  rownames(df) = c("Out1", "Out2", "Out3")
  colnames(df) = c("Tar1", "Tar2", "Tar3")
  pdf(file=paste("conf", gsub("/.", "_", r), ".pdf",sep=""),width=8.97,height=5.76)
  heatmap(as.matrix(df), Rowv=NA, main = r, Colv=NA, col = my_palette, scale="none", margins=c(5,10))
  dev.off()
}