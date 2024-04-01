
value_data <- read.delim("/Users/kevindu/Desktop/Employment/Multiagent Snake Research/kevindu@login.rc.fas.harvard.edu/multiagent_snake/valueLoss.out", sep=',', header = FALSE)
policy_data <- read.delim("/Users/kevindu/Desktop/Employment/Multiagent Snake Research/kevindu@login.rc.fas.harvard.edu/multiagent_snake/policyLoss.out", sep=',', header = FALSE)
norm_data <- read.delim("/Users/kevindu/Desktop/Employment/Multiagent Snake Research/kevindu@login.rc.fas.harvard.edu/multiagent_snake/norm.out", sep=',', header = FALSE)
N = length(value_data)

value_data <- value_data[-N]
policy_data <- policy_data[-N]
norm_data <- norm_data[-N]
N = N-1

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

plot(1:N, value_data, main="value loss")
# lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(value_data, NUM_AVG), col="red")
plot(1:N, policy_data, main="policy loss")
# lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(policy_data, NUM_AVG), col="red")
plot(1:N, norm_data, main="grad norm")
# lines(1:(N/NUM_AVG) * NUM_AVG, consec_avg(norm_data, NUM_AVG), col="red")
