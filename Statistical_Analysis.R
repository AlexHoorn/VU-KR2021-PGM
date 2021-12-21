#### Probabilistic Graphical Models
##
## Knowledge Representation
##
## Plots and Statistical Analysis of Experiments 
##
##
## last mod: 20.12.21

# working directory
# setwd("C:/Users/oleh/Documents/Uni/13_Semester/Knowledge_Representation/Assignment_2/KR21_project2/")



####### Input Data #################

# f.names <- dir(pattern = ".csv")



## Asusuming all exponential models with increasing network size


### heuristics of ordering

dat <- read.table("ordering_results.csv", header = TRUE, dec = ".", sep = ",")
# View(dat)

dat <- dat[,c("nodes", "experiment", "runtime")]
dat$experiment <- factor(dat$experiment)

heu <- levels(dat$experiment)
cols <- c("darkgreen", "darkred", "darkblue")

# 5 to 505 in steps of 10, each time 10 times each
# xtabs(~ experiment, dat) 510 in total for each heuristic

## analysis:

# ancova:
m1 <- lm(log(runtime) ~ nodes, data = dat)
#summary(m1)
m2 <- lm(log(runtime) ~ nodes + experiment, data = dat)
#summary(m2)
m3 <- lm(log(runtime) ~ nodes*experiment, data = dat)
#summary(m3)

anova(m1,m2,m3)

# Ancova allows for different slopes, all three heuristics differ significantly (p < .001)
# with respect to slope and intercept
# has to be mentioned: whereas mindeg seems linear, order random and order minfill seems to
# increase quadratic / exponentially.


lm_coef <- coef(m3)

## plot
pdf("plots/heuristics.pdf", width = 10, height = 6, pointsize = 10)
par(mgp = c(2.5, 0.7, 0), mar = c(4, 4, 1, 1) + 0.1)
plot(runtime ~ nodes, data = dat[dat$experiment == heu[1],], type = "p", col = cols[1],
     ylab = "Runtime (sec)", xlab = "Network size", ylim = c(0,0.38))
abline(h = 0, col = "grey", lty = "dotdash")
segments(0, -1, 0, 0.3, lty = "dotdash", col = "grey")
points(runtime ~ nodes, data = dat[dat$experiment == heu[2],], type = "p", col = cols[2])
points(runtime ~ nodes, data = dat[dat$experiment == heu[3],], type = "p", col = cols[3])
lines(c(0:600), exp(lm_coef[1])*exp(lm_coef[2]*c(0:600)), col = cols[1])
lines(c(0:600), exp(lm_coef[1] + lm_coef[3])*exp((lm_coef[2] + lm_coef[5])*c(0:600)), col = cols[2])
lines(c(0:600), exp(lm_coef[1] + lm_coef[4])*exp((lm_coef[2] + lm_coef[6])*c(0:600)), col = cols[3])
#abline(lm(runtime ~ nodes, data = dat[dat$experiment == heu[1],]), col = cols[1])
#abline(lm(runtime ~ nodes, data = dat[dat$experiment == heu[2],]), col = cols[2])
#abline(lm(runtime ~ nodes, data = dat[dat$experiment == heu[3],]), col = cols[3])
legend("topleft", legend = heu, col = cols, 
       lty = 1, cex=1, lwd = 2, box.col = "white")
box()
dev.off()





### MAP / MPE

dat <- read.table("map_mpe_results.csv", header = TRUE, dec = ".", sep = ",")
# View(dat)

dat$task <- factor(substr(dat$experiment, 0, 3))
dat$heu <- factor(substr(dat$experiment, 5, 13))

task <- levels(dat$task)
heu <- levels(dat$heu)
cols <- c("darkgreen", "darkred", "darkblue")


# some descriptive data:

#table(dat$task)
# map  mpe 
# 1920  672 

### Timeouts in MAP

# from 2 to 66 -> 65 group, each 10 times
# MAP: in total 660
# random: 645 (5) -> 650
# minfill: 644 (6) -> 650
# mindeg: 631 (19) -> 650


### Timeouts in MPE

# from 2 to 31 --> 30, each 10 times
# # MPE: in total 155
# random: 225 (75) -> 300
# mindeg: 226 (74) -> 300
# minfill: 221 (79) -> 300





# how did we come to this number? (MAP and MPE)

# too generate more realistic networks, they were adjusted in advance
# for instance, there are no unconnected nodes


#xtabs(~ task + heu, dat)
#                heu
# task  mindeg minfill random
# map    631     644    645
# mpe    226     221    225




### MAP


## analysis
m1 <- lm(log(runtime) ~ nodes, dat = dat[dat$task == task[1],])
# summary(m1)
m2 <- lm(log(runtime) ~ nodes + heu, dat = dat[dat$task == task[1],])
# summary(m2)
m3 <- lm(log(runtime) ~ nodes * heu, dat = dat[dat$task == task[1],])
# summary(m3)

anova(m1,m2,m3)
# Whereas the first model shows a significant impact of the network size on the runtime
# (t(1) = 7.72, p < .001), neither a model that allows for an additional effect of the heuristics
# (F(2,146) = 0.47, p = .625) nor a model that allows for an additional effect and an 
# interaction of the heuristic with the slope (F(2, 144) = 0.27, p = 0.766) can explain 
# significantly more variance. 


lm_coef <- coef(m3)

## plots for MAP und MPE, splitted?
## plot
pdf("plots/MAP.pdf", width = 10, height = 6, pointsize = 10)
par(mgp = c(2.5, 0.7, 0), mar = c(4, 4, 1, 1) + 0.1)
plot(runtime ~ jitter(nodes), data = dat[dat$task == task[1] & dat$heu == heu[1],], type = "p", col = cols[1],
     ylab = "Runtime (sec)", xlab = "Network size", ylim = c(0,10), xlim = c(0,45))
abline(h = 0, col = "grey", lty = "dotdash")
segments(0, -10, 0, 8.5, lty = "dotdash", col = "grey")
points(runtime ~ jitter(nodes), data = dat[dat$task == task[1] & dat$heu == heu[2],], type = "p", col = cols[2])
points(runtime ~ jitter(nodes), data = dat[dat$task == task[1] & dat$heu == heu[3],], type = "p", col = cols[3])
#abline(lm(runtime ~ nodes, data = dat[dat$task == task[1] & dat$heu == heu[1],]), col = cols[1])
#abline(lm(runtime ~ nodes, data = dat[dat$task == task[1] & dat$heu == heu[2],]), col = cols[2])
#abline(lm(runtime ~ nodes, data = dat[dat$task == task[1] & dat$heu == heu[3],]), col = cols[3])
lines(c(0:50), exp(lm_coef[1])*exp(lm_coef[2]*c(0:50)), col = cols[1])
lines(c(0:50), exp(lm_coef[1] + lm_coef[3])*exp((lm_coef[2] + lm_coef[5])*c(0:50)), col = cols[2])
lines(c(0:50), exp(lm_coef[1] + lm_coef[4])*exp((lm_coef[2] + lm_coef[6])*c(0:50)), col = cols[3])
legend("topleft", legend = heu, col = cols, 
       lty = 1, cex=1, lwd = 2, box.col = "white")
box()
dev.off()




### MPE

## analysis
m1 <- lm(log(runtime) ~ nodes, dat = dat[dat$task == task[2],])
# summary(m1)
m2 <- lm(log(runtime) ~ nodes + heu, dat = dat[dat$task == task[2],])
# summary(m2)
m3 <- lm(log(runtime) ~ nodes * heu, dat = dat[dat$task == task[2],])
# summary(m3)

anova(m1,m2,m3)
# Whereas the first model shows a significant impact of the network size on the runtime
# (t(1) = 7.72, p < .001), neither a model that allows for an additional effect of the heuristics
# (F(2,146) = 0.47, p = .625) nor a model that allows for an additional effect and an 
# interaction of the heuristic with the slope (F(2, 144) = 0.27, p = 0.766) can explain 
# significantly more variance. 

lm_coef <- coef(m3)


## plot
pdf("plots/MPE.pdf", width = 10, height = 6, pointsize = 10)
par(mgp = c(2.5, 0.7, 0), mar = c(4, 4, 1, 1) + 0.1)
plot(runtime ~ jitter(nodes), data = dat[dat$task == task[2] & dat$heu == heu[1],], type = "p", col = cols[1],
     ylab = "Runtime (sec)", xlab = "Network size", ylim = c(0,120), xlim = c(0, 25))
abline(h = 0, col = "grey", lty = "dotdash")
segments(0, -100, 0, 100, lty = "dotdash", col = "grey")
points(runtime ~ jitter(nodes), data = dat[dat$task == task[2] & dat$heu == heu[2],], type = "p", col = cols[2])
points(runtime ~ jitter(nodes), data = dat[dat$task == task[2] & dat$heu == heu[3],], type = "p", col = cols[3])
lines(c(0:30), exp(lm_coef[1])*exp(lm_coef[2]*c(0:30)), col = cols[1])
lines(c(0:30), exp(lm_coef[1] + lm_coef[3])*exp((lm_coef[2] + lm_coef[5])*c(0:30)), col = cols[2])
lines(c(0:30), exp(lm_coef[1] + lm_coef[4])*exp((lm_coef[2] + lm_coef[6])*c(0:30)), col = cols[3])
legend("topleft", legend = heu, col = cols, 
       lty = 1, cex=1, lwd = 2, box.col = "white")
box()
dev.off()






## analysis: comparing MAP and MPE

## analysis
m1 <- lm(log(runtime) ~ nodes, dat = dat)
# summary(m1)
m2 <- lm(log(runtime) ~ nodes + task, dat = dat)
# summary(m2)
m3 <- lm(log(runtime) ~ nodes * task, dat = dat)
# summary(m3)

anova(m1,m2,m3)

lm_coef <- coef(m3)

pdf("plots/MAP_MPE.pdf", width = 10, height = 6, pointsize = 10)
par(mgp = c(2.5, 0.7, 0), mar = c(4, 4, 1, 1) + 0.1)
plot(runtime ~ jitter(nodes), data = dat[dat$task == task[1],], type = "p", col = cols[1],
     ylab = "Runtime (sec)", xlab = "Network size", ylim = c(0,115), xlim = c(0, 25))
abline(h = 0, col = "grey", lty = "dotdash")
segments(0, -100, 0, 100, lty = "dotdash", col = "grey")
points(runtime ~ jitter(nodes), data = dat[dat$task == task[2],], type = "p", col = cols[2])
lines(c(0:30), exp(lm_coef[1])*exp(lm_coef[2]*c(0:30)), col = cols[1])
lines(c(0:30), exp(lm_coef[1] + lm_coef[3])*exp((lm_coef[2] + lm_coef[4])*c(0:30)), col = cols[2])
legend("topleft", legend = task, col = cols, 
       cex=1, lwd = 2, box.col = "white")
box()
dev.off()



### exploratory analysis of several metrics

## multiple (exponential) regression

# not checked for multicollinearity (which quite sure exists tho)

m1 <- lm(log(runtime) ~ nodes * task + roots + leaves + mean_edges, dat = dat)
summary(m1)

