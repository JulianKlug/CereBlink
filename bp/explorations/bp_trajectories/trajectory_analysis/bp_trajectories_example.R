# Ref: https://github.com/philips-labs/demo-clustering-longitudinal-data/blob/main/analysis_gbtm.R
# Paper: https://arxiv.org/pdf/2111.05469

library(lcmm)
library(splines)
library(magrittr)
library(assertthat)
library(data.table)
library(ggplot2)
library(scales)
library(matrixStats)

source(./'plotting.R')
source(./'utils.R')

generate_osa_data = function(patients=500,
                             times=seq(1, 365, by=1),
                             nAggr=14,
                             props=c(GU=.24, SI=.13, SD=.14, VU=.17, OA=.08, ED=.13, NU=.11),
                             # N=c(354, 344, 280, 299, 106, 55, 10),
                             # sd.N=c(31, 49, 68, 55, 64, 30, 4),
                             dropoutTimes=c(Inf, Inf, Inf, Inf, Inf, 80, 20),
                             sd.dropoutTimes=c(0, 0, 0, 0, 0, 30, 10),
                             attemptProbs=c(354, 344, 280, 299, 106, 55, 14) / pmin(dropoutTimes, 365), # based on median
                             intercepts=c(GU=6.6, SI=5.8-1, SD=6.1, VU=4.9-.5, OA=3.2, ED=4.0, NU=2.5),
                             sd.intercepts=c(GU=.81, SI=1.6-.1, SD=.95, VU=1.3, OA=1.6, ED=1.6, NU=1.4) * .667,
                             slopes=c(GU=0, SI=.0058*3, SD=-.0038*5, VU=.0004*24, OA=-.003, ED=-.0014, NU=-.015),
                             sd.slopes=c(GU=.0016, SI=.0031/2, SD=.0027/2, VU=.0032*0, OA=.0091, ED=.01, NU=.01),
                             quads=c(GU=0, SI=-.00003, SD=.00003, VU=-.00003, OA=0, ED=-.0001, NU=-.0001),
                             sd.quads=c(GU=0, SI=0, SD=0, VU=0, OA=0, ED=0, NU=0),
                             vars=c(GU=2.0, SI=3.6, SD=3.2, VU=3.4, OA=3.6, ED=5.0, NU=3.0),
                             sd.vars=c(.82, 1.3, .85, 1.2, 1.8, 2.6, 1.7),
                             autocors=c(GU=.056, SI=.11, SD=.073, VU=.048, OA=.006, ED=-.044, NU=-.31),
                             groupNames=c('Good users', 'Slow improvers', 'Slow decliners', 'Variable users', 'Occasional attempters', 'Early drop-outs', 'Non-users'),
                             missing=FALSE,
                             seed=1) {
    set.seed(seed)
    groupCounts = floor(patients * props)
    incrIdx = order((patients * props) %% 1) %>%
        head(patients - sum(groupCounts))
    groupCounts[incrIdx] = groupCounts[incrIdx] + 1 # increment the groups that were closest to receiving another patient
    assert_that(sum(groupCounts) == patients)
    groupNames = factor(groupNames, levels=groupNames)
    
    # generate patient coefficients
    groupCoefs = data.table(Group=groupNames, Patients=groupCounts,
                            TDrop=dropoutTimes, Sd.TDrop=sd.dropoutTimes,
                            AProb=attemptProbs,
                            Int=intercepts, Sd.Int=sd.intercepts,
                            Slope=slopes, Sd.Slope=sd.slopes,
                            Quad=quads, Sd.Quad=sd.quads,
                            Var=vars, Sd.Var=sd.vars,
                            R=autocors)
    assert_that(nrow(groupCoefs) == 7)
    patCoefs = groupCoefs[, .(TDrop=rnorm(Patients, TDrop, Sd.TDrop) %>% round %>% pmax(7),
                              AProb=AProb,
                              Intercept=rnorm(Patients, Int, Sd.Int) %>% pmax(0),
                              Slope=rnorm(Patients, Slope, Sd.Slope),
                              Quad=rnorm(Patients, Quad, Sd.Quad),
                              Variance=rnorm(Patients, Var, Sd.Var) %>% pmax(.75),
                              R=rep(R, Patients)), by=Group]
    
    # generate patient measurements
    genTs = function(N, Intercept, Slope, Quad, Variance, R, AProb, TDrop, ...) {
        y = as.numeric(Intercept + times * Slope + times^2 * Quad + arima.sim(list(ar=R), n=length(times), sd=sqrt(Variance)))
        
        skipMask = !rbinom(length(times), size=1, prob=AProb)
        y[skipMask] = 0
        
        if(missing) {
            obsMask = times <= TDrop
            list(Time=times[obsMask], Usage=pmax(y[obsMask], 0))
        } else {
            y[times > TDrop] = 0
            list(Time=times, Usage=pmax(y, 0))
        }
    }
    
    patNames = paste0('P', 1:patients)
    alldata = patCoefs[, do.call(genTs, .SD), by=.(Group, Id=factor(patNames, levels=patNames))] %>%
        setkey(Id, Time)
    
    # generate group trajectories
    groupTrajs = groupCoefs[, .(Time=times,
                                Usage=(pmax(Int + times * Slope + times^2 * Quad, 0) * AProb) %>% ifelse(times > TDrop, 0, .)
    ), by=Group]
    
    setattr(alldata, 'patCoefs', patCoefs)
    setattr(alldata, 'groupCoefs', groupCoefs)
    setattr(alldata, 'groupTrajs', groupTrajs)
    
    # Extra step: downsampling
    if(is.finite(nAggr)) {
        alldata = transformToAverage(alldata, binSize=nAggr)
    }
    return(alldata[])
}


data = generate_osa_data() %>%
    .[, Id := as.integer(Id)] %>%
    .[, NormTime := (Time - min(Time)) / (max(Time) - min(Time))]

makeGbtmCall = function(k) {
    substitute(
        hlme(fixed=Usage ~ NormTime + I(NormTime^2),
             mixture=~NormTime + I(NormTime^2),
             random=~-1,
             subject='Id', ng=k, data=data),
        env=list(k=k)
    )
}

computeHlmeTrajectories = function(model) {
    times = sort(unique(data$Time))
    normTimes = sort(unique(data$NormTime))
    predictY(model, newdata=data.frame(NormTime=normTimes))$pred %>%
        data.table(Time=times) %>%
        melt(id.vars='Time', value.name='Usage', variable.name='Group') %>%
        .[, Group := factor(Group, levels=paste0('Ypred_class', 1:model$ng), labels=LETTERS[1:model$ng])] %>%
        .[]
}

# Single-group analysis ####
mod00 = hlme(fixed=Usage ~ 1, random=~1, subject='Id', ng=1, data=data)
summary(mod00)
residuals(mod00) %T>% qqnorm %>% qqline

mod11 = hlme(fixed=Usage ~ NormTime, random=~NormTime, subject='Id', ng=1, data=data)
summary(mod11)
residuals(mod11) %T>% qqnorm %>% qqline

mod22 = hlme(fixed=Usage ~ poly(NormTime, 2, raw=TRUE), random=~poly(NormTime, 2, raw=TRUE), subject='Id', ng=1, data=data)
summary(mod22)
residuals(mod22) %T>% qqnorm %>% qqline

mod33 = hlme(fixed=Usage ~ poly(NormTime, 3, raw=TRUE), random=~poly(NormTime, 3, raw=TRUE), subject='Id', ng=1, data=data)
summary(mod33)
residuals(mod33) %T>% qqnorm %>% qqline

modbs = hlme(fixed=Usage ~ bs(NormTime), random=~bs(NormTime), subject='Id', ng=1, data=data)
summary(modbs)
residuals(modbs) %T>% qqnorm %>% qqline

# Estimation ####
gbtms = list()
# gbtms = readRDS('save/gbtm.rds')
gbtms[['1']] = hlme(fixed=Usage ~ NormTime + I(NormTime^2), random=~-1, subject='Id', ng=1, data=data)

fitGbtm = function(k) {
    start = Sys.time()
    model = do.call(gridsearch, list(m=makeGbtmCall(k), rep=20, maxiter=1, minit=gbtms[['1']]))
    model$runTime = Sys.time() - start
    return(model)
}

gbtms[['2']] = fitGbtm(2)
gbtms[['3']] = fitGbtm(3)
gbtms[['4']] = fitGbtm(4)
gbtms[['5']] = fitGbtm(5)
gbtms[['6']] = fitGbtm(6)
gbtms[['7']] = fitGbtm(7)
gbtms[['8']] = fitGbtm(8)
# saveRDS(gbtms, file='save/gbtm.rds')

# Solutions ####
plotMetric(sapply(gbtms, '[[', 'BIC'), as.integer(names(gbtms)), 'BIC')
# ggsave('save/gbtm_bic.pdf', width=bicPlotSize[1], height=bicPlotSize[2], units='cm')

# Assess the best solution ####
k = 4
bestGbtm = gbtms[[as.character(k)]]

pp = bestGbtm$pprob[paste0('prob', 1:k)] %>%
    as.matrix() %>%
    set_colnames(LETTERS[1:k])
groupProps = colMeans(pp) %T>% print()

computeHlmeTrajectories(bestGbtm) %>% plotGroupTrajectories(groupProps)
# ggsave('save/gbtm_groups.pdf', width=groupPlotSize[1], height=groupPlotSize[2], units='cm')


appa(pp)
relativeEntropy(pp)

#’ Assign each subject in newdata to its most likely latent class
#’
#’ @param newdata  A data.frame containing at least the outcome and covariates
#’                 (here: Usage, NormTime) plus the subject ID column.
#’ @param model    A fitted hlme model (your bestGbtm object).
#’ @param subject  Name of the ID‐column in newdata (default: "Id").
#’
#’ @return A data.frame with one row per subject and two columns:
#’         the subject ID and the assigned cluster (class).
#’
#’ @examples
#’ # suppose you have new_osa_data from generate_osa_data()
#’ # cluster_map <- classify_new_data(new_osa_data, model = bestGbtm)
#’
classify_new_data <- function(newdata,
                              model   = bestGbtm,
                              subject = "Id") {
    # make sure lcmm is loaded
    if (!"lcmm" %in% .packages()) library(lcmm)
    
    # compute posterior class‐membership for each subject in newdata
    # returns a matrix with columns: <subject>, "class", "prob1", ..., "probK"
    pmat <- predictClass(model,
                         newdata = newdata,
                         subject = subject)
    
    # coerce to data.frame for easy subsetting
    pdf <- as.data.frame(pmat, stringsAsFactors = FALSE)
    
    # keep only one row per subject (they’re all the same within each subject)
    subj_class <- unique(pdf[, c(subject, "class")])
    
    # rename columns
    names(subj_class) <- c(subject, "Cluster")
    
    return(subj_class)
}

cluster_map <- classify_new_data(data[data$Id == 300], model = bestGbtm)

