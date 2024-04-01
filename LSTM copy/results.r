
data <- read.delim("/Users/kevindu/Desktop/Coding/Tests:experiments/LSTM/results.out", sep=',', header = FALSE)

plot(1:length(data[,2]), data[,2], type='l')