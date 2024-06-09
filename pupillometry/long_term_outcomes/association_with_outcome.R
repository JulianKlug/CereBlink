library(MASS)

path <- '/Users/jk1/Downloads/joined_df.csv'
df <- read.csv(path)

df$outcome <- factor(df$mRS_FU_1y, levels = c(0, 1, 2, 3, 4, 5, 6), ordered = TRUE)

m <- polr(outcome ~ inter_eye_min_NPI + inter_eye_min_CV
          + Age + WFNS + Fisher_Score,
          data = df, Hess=TRUE)

summary(m)

table = coef(summary(m))
p = pnorm(abs(table[, "t value"]),
lower.tail = FALSE) * 2
round(p, 4)

