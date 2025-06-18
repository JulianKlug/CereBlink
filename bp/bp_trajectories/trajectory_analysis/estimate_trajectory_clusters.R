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

# Estimation ####
gbtms = list()
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


