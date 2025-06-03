options(repos = c(CRAN = "https://cran.r-project.org"))

# Install ordinal package if not installed
if (!require(ordinal)) {
    install.packages("ordinal")
}
library(ordinal)

path <- "/Users/jk1/Downloads/first_24h_df_hourly_cat.csv"
df <- read.csv(path, header = TRUE)

df$response <- factor(df$mrs_1y, ordered=TRUE)
df$mitteldruck <- as.numeric(df$mitteldruck)
# pnr is unique patient number, is a string
df$pNr <- as.factor(df$pNr)

df$mitteldruck <- scale(df$mitteldruck)

model <- clmm(response ~ mitteldruck + (1 | pNr), data=df, control = clmm.control(maxIter = 1000))
summary(model)