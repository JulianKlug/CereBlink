library(pspline)
library(survival)

univariate_data_path <- '/Users/jk1/Downloads/univariate_predictor_df.csv'
multivariate_data_path <- '/Users/jk1/Downloads/multivariate_predictor_df.csv'
univariate_data <- read.csv(univariate_data_path)
multivariate_data <- read.csv(multivariate_data_path)

# Univariate Cox PH
## create the survival object
surv.death <- Surv(univariate_data$follow_up_time, univariate_data$Death)
## fit survival model
fit.death <- coxph(surv.death ~ pspline(univariate_data$pCO2, df=3), data=univariate_data)
## print the summary
summary(fit.death)

## get predicted values for fitted spline
predicted <- predict(fit.death , type = "terms" , se.fit = TRUE , terms = 1)
## plot the spline
plot(univariate_data$pCO2, exp(predicted$fit), type="n", xlab="pCO2", ylab="Hazard Ratio", main="Hazard Ratio of pCO2")
lines( sm.spline(univariate_data$pCO2 , exp(predicted$fit)) , col = "red" , lty = 1 )
lines( sm.spline(univariate_data$pCO2 , exp(predicted$fit + 1.96 * predicted$se)) , col = "orange" , lty = 2 )
lines( sm.spline(univariate_data$pCO2 , exp(predicted$fit - 1.96 * predicted$se)) , col = "orange" , lty = 2 )


# Multivariate Cox PH
## create the survival object
surv.death <- Surv(multivariate_data$follow_up_time, multivariate_data$Death)
## fit survival model
fit.death <- coxph(surv.death ~ pspline(multivariate_data$pCO2, df=2)
                    + multivariate_data$Age + multivariate_data$Sex + multivariate_data$mRS_before_ictus
                   + multivariate_data$GCS_admission + multivariate_data$WFNS + multivariate_data$Intubated_on_admission_YN
                   + multivariate_data$HTN + multivariate_data$DM + multivariate_data$pO2 + multivariate_data$mitteldruck,
                     data=multivariate_data)
## print the summary
summary(fit.death)

## get predicted values for fitted spline
predicted <- predict(fit.death , type = "terms" , se.fit = TRUE , terms = 1)
## plot the spline
# set ylimits(0, 5)
plot(multivariate_data$pCO2, exp(predicted$fit), type="n", xlab="pCO2", ylab="Hazard Ratio", main="Hazard Ratio of pCO2", ylim=c(0,2))
lines( sm.spline(multivariate_data$pCO2 , exp(predicted$fit)) , col = "red" , lty = 1 )
lines( sm.spline(multivariate_data$pCO2 , exp(predicted$fit + 1.96 * predicted$se)) , col = "orange" , lty = 2 )
lines( sm.spline(multivariate_data$pCO2 , exp(predicted$fit - 1.96 * predicted$se)) , col = "orange" , lty = 2 )
