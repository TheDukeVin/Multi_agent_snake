
training_data <- read.delim("/Users/kevindu/Desktop/Employment/Multiagent Snake Research/kevindu@login.rc.fas.harvard.edu/multiagent_snake/training200.out", sep=',', header = FALSE)
N = length(training_data[1,])

# EvalPeriod = 1000
# 
NUM_AVG = 50

consec_avg <- function(lst, N){
  avgs <- integer(length(lst) / N)
  for(i in 0:(length(lst) / N - 1)){
    x = 0
    count = 0
    for(j in 1:N){
      num = as.numeric(lst[i*N + j])
      if(num >= 0){
        x = x + num
        count = count + 1
      }
    }
    avgs[i+1] = x / count
  }
  avgs
}

# f <- function(x){
#   (x > 10) - (x < -10)
# }
# 
# for(i in 1:(length(x) / EvalPeriod + 1) ){
#   range = x[(EvalPeriod*(i-1) + 1) : (EvalPeriod * i)]
#   plot(1:EvalPeriod, range, main=paste("Period", i), ylab="Score", xlab="Game number")
#   lines((1:(EvalPeriod/NUM_AVG)) * NUM_AVG, consec_avg(range, NUM_AVG), col="red")
# }

plot(1:N, training_data[1,], main="value loss")
lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(training_data[1,], NUM_AVG), col="red")
plot(1:N, training_data[2,], main="policy loss")
lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(training_data[2,], NUM_AVG), col="red")
plot(1:N, training_data[3,], main="grad norm")
lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(training_data[3,], NUM_AVG), col="red")
