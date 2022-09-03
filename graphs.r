library(graphics)
library(grid)

data <- read.delim("/Users/kevindu/Desktop/Employment/multiagent_snake/multiagent_snake/features.txt", sep='\t', header = FALSE)

nums <- data[1,]

N = length(nums)

NUM_AVG = 50

consec_avg <- function(lst, N){
  avgs <- integer(length(lst) / N)
  for(i in 0:(length(lst) / N - 1)){
    x = 0
    count = 0
    for(j in 1:N){
      num = lst[i*N + j]
      if(num >= 0){
        x = x + num
        count = count + 1
      }
    }
    avgs[i+1] = x / count
  }
  avgs
}

M = floor(N / NUM_AVG)

par(las=1, mar = c(3, 3, 1, 1))
plot(1:M * NUM_AVG, consec_avg(nums, NUM_AVG), type="l",
     xlim=c(1,N), ylim=c(50,100),
     xlab="", ylab="",
     xaxt="n", yaxt="n", col="red")
axis(2, mgp=c(3, .5, 0))
axis(1, mgp=c(3, .3, 0))
title(ylab="Average score", line=2, cex.lab=1)
title(xlab="Game Number", line=1.5, cex.lab=1)

#plot(c(), c(), xlim=c(1,N), ylim=c(50,100), xlab="Game number", ylab="Average score")
#lines(1:length(avgs) * NUM_AVG, avgs, col="red")

lines(c(0, N), c(59.09, 59.09), col="black", lty=2)
lines(c(0, N), c(100, 100), col="gray30", lty=3)

legend(2500, 85, legend=c("AlphaZero", "Naive tree search", "Maximum score"),
       col=c("red", "black", "gray30"), lty=1:3, cex=0.8)


plot(1:M * NUM_AVG, consec_avg(nums == 100, NUM_AVG), type="l",
     xlim=c(1,N), ylim=c(0,1),
     xlab="", ylab="",
     xaxt="n", yaxt="n", col="blue")
axis(2, mgp=c(3, .5, 0))
axis(1, mgp=c(3, .3, 0))
title(ylab="Win rate", line=2, cex.lab=1)
title(xlab="Game Number", line=1.5, cex.lab=1)

# plot(c(), c(), xlim=c(1,N), ylim=c(0,1), xlab="Game number", ylab="Win rate")
# lines(1:length(comps) * NUM_AVG, comps, col="blue")

layout(matrix(1:5, ncol = 1), widths = 1, heights = c(1.5,1,1,1.5,1.5), respect = FALSE)
par(mar = c(0, 4.1, 4.1, 2.1))
plot(1:M * NUM_AVG, consec_avg(data[2,], NUM_AVG), ylab = "Wall-hug states", type = 'l', xaxt = 'n', main = 'Learned Features')
par(mar = c(0, 4.1, 0, 2.1))
plot(1:M * NUM_AVG, consec_avg(data[3,], NUM_AVG), ylab = "Visibility", ylim=c(0.7,1), type = 'l', xaxt = 'n')
lines(1:M * NUM_AVG, consec_avg(data[4,], NUM_AVG), col="red")
legend(4500, 0.8, legend=c("Apple visibility", "Tail visibility"),
       col=c("black", "red"), lty = c(1,1), cex=0.8)
par(mar = c(0, 4.1, 0, 2.1))
plot(1:M * NUM_AVG, consec_avg(data[5,], NUM_AVG), ylab = "Distance", ylim=c(4.7,6.5), type = 'l', xaxt = 'n')
lines(1:M * NUM_AVG, consec_avg(data[6,], NUM_AVG), col="red")
par(mar = c(0, 4.1, 0, 2.1))
legend(4500, 6.5, legend=c("L1 distance", "Dynamic distance"),
       col=c("black", "red"), lty = c(1,1), cex=0.8)
par(mar = c(0, 4.1, 0, 2.1))
plot(1:M * NUM_AVG, consec_avg(data[7,], NUM_AVG), ylab = "Connected Components", type = 'l', xaxt = 'n')
par(mar = c(4.1, 4.1, 0, 2.1))
plot(1:M * NUM_AVG, consec_avg(data[10,], NUM_AVG), ylab = "Turns", type = 'l', xlab = "Game Number")


# plot features

# plot(c(), c(), xlim=c(1,N), ylim=c(0.06,0.16), xlab="Game number", ylab="Wall-hug states")
# lines(1:M * NUM_AVG, consec_avg(data[2,], NUM_AVG))
# 
# plot(c(), c(), xlim=c(1,N), ylim=c(0.7,1), xlab="Game number", ylab="Apple and tail visibility")
# lines(1:M * NUM_AVG, consec_avg(data[3,], NUM_AVG))
# lines(1:M * NUM_AVG, consec_avg(data[4,], NUM_AVG), col="red")
# 
# legend(4000, 0.82, legend=c("Apple visibility", "Tail visibility"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# plot(c(), c(), xlim=c(1,N), ylim=c(4.7,6.5), xlab="Game number", ylab="L1 and Dynamic distance")
# lines(1:M * NUM_AVG, consec_avg(data[5,], NUM_AVG))
# lines(1:M * NUM_AVG, consec_avg(data[6,], NUM_AVG), col="red")
# 
# legend(3000, 6.3, legend=c("L1 distance", "Dynamic distance"),
#        col=c("black", "red"), lty = c(1,1), cex=0.8)
# 
# plot(c(), c(), xlim=c(1,N), ylim=c(1.6, 2.4), xlab="Game number", ylab="Connected components")
# lines(1:M * NUM_AVG, consec_avg(data[7,], NUM_AVG))

# plot(c(), c(), xlim=c(1,N), ylim=c(0.5,1), xlab="Game number", ylab="L1 movement")
# lines(1:M * NUM_AVG, consec_avg(data[8,], NUM_AVG))
# 
# plot(c(), c(), xlim=c(1,N), ylim=c(0.5,1), xlab="Game number", ylab="Dynamic distance movement")
# lines(1:M * NUM_AVG, consec_avg(data[9,], NUM_AVG))

# plot(1:M * NUM_AVG, consec_avg(data[10,], NUM_AVG), type="l",
#      xlim=c(1,N), ylim=c(0.16,0.25),
#      xlab="", ylab="",
#      xaxt="n", yaxt="n")
# axis(2, mgp=c(3, .5, 0))
# axis(1, mgp=c(3, .3, 0))
# title(ylab="Turns", line=2.3, cex.lab=1)
# title(xlab="Game Number", line=1.5, cex.lab=1)
