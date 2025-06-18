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

source('bp/explorations/bp_trajectories/trajectory_analysis/plotting.R')
source('bp/explorations/bp_trajectories/trajectory_analysis/utils.R')

data_path <- "/Users/jk1/Downloads/hourly_bp_df.csv"
data <- fread(data_path) %>%
    .[, pNr := as.integer(pNr)]

# restrict data to relative_timeBd_hours < 22
data = data[relative_timeBd_hours < 22*24]


makeGbtmCall = function(k) {
    substitute(
        hlme(fixed=mitteldruck ~ relative_timeBd_hours + I(relative_timeBd_hours^2) + I(relative_timeBd_hours^3) + I(relative_timeBd_hours^4),
             mixture=~relative_timeBd_hours + I(relative_timeBd_hours^2) + I(relative_timeBd_hours^3) + I(relative_timeBd_hours^4),
             random=~-1,
             subject='pNr', ng=k, data=data),
        env=list(k=k)
    )
}

computeHlmeTrajectories = function(model) {
    times = sort(unique(data$relative_timeBd_hours))
    relativeTime = sort(unique(data$relative_timeBd_hours))
    predictY(model, newdata=data.frame(relative_timeBd_hours=relativeTime))$pred %>%
        data.table(Time=times) %>%
        melt(id.vars='Time', value.name='mitteldruck', variable.name='Group') %>%
        .[, Group := factor(Group, levels=paste0('Ypred_class', 1:model$ng), labels=LETTERS[1:model$ng])] %>%
        .[]
}


# Estimation ####
gbtms = list()
# gbtms = readRDS('/Users/jk1/Downloads/hourly_order4_gbtm.rds')
gbtms[['1']] = hlme(
    fixed=mitteldruck ~ relative_timeBd_hours + I(relative_timeBd_hours^2) + I(relative_timeBd_hours^3) + I(relative_timeBd_hours^4),
    random=~-1, 
    subject='pNr', 
    ng=1, 
    data=data)

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
saveRDS(gbtms, file='/Users/jk1/Downloads/hourly_order4_clip21d_gbtm.rds')

# Solutions ####
plotMetric(sapply(gbtms, '[[', 'BIC'), as.integer(names(gbtms)), 'BIC')
ggsave('/Users/jk1/Downloads/hourly_order4_clip21d_gbtm_bic.pdf')

# Assess the best solution ####
k = 6
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

cluster_map <- classify_new_data(data, model = bestGbtm, subject = "pNr")
# save cluster_map
# fwrite(cluster_map, file='/Users/jk1/Downloads/hourly_order4_clip21d_gbtm_cluster_map.csv')

