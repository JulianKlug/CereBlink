library(splines)
library(survival)
options(digits=3)
library(tidyverse)

univariate_data_path <- '/Users/jk1/Downloads/univariate_predictor_df.csv'
multivariate_data_path <- '/Users/jk1/Downloads/multivariate_predictor_df.csv'
univariate_data <- read.csv(univariate_data_path)
multivariate_data <- read.csv(multivariate_data_path)

# drop follow_up_time column by name
univariate_data <- univariate_data[, !names(univariate_data) %in% c("follow_up_time")]
multivariate_data <- multivariate_data[, !names(multivariate_data) %in% c("follow_up_time")]

# make a pairs plot of the multi-variate data
pairs(multivariate_data, col=as.factor(multivariate_data$Death))

# Univariate Logistic Regression
univariate_glm <- glm(Death ~ pCO2, data=univariate_data, family='binomial')
summary(univariate_glm)
univariate_spline_glm <- glm(Death ~ ns(pCO2, df=3), data=univariate_data, family='binomial')
summary(univariate_spline_glm)

# Multivariate Logistic Regression
multivariate_glm <- glm(Death ~ pCO2 + Age + Sex + mRS_before_ictus + GCS_admission + WFNS
  + Intubated_on_admission_YN + HTN + DM + pO2 + mitteldruck, data=multivariate_data, family='binomial')
summary(multivariate_glm)
backstep_multivariate_glm <- step(multivariate_glm, direction="backward")
summary(backstep_multivariate_glm)

multivariate_spline_glm <- glm(Death ~ ns(pCO2, df=3) + Age + Sex + mRS_before_ictus + GCS_admission + WFNS
  + Intubated_on_admission_YN + HTN + DM + pO2 + mitteldruck, data=multivariate_data, family='binomial')
summary(multivariate_spline_glm)
backstep_multivariate_spline_glm <- step(multivariate_spline_glm, direction="backward")
summary(backstep_multivariate_spline_glm)
