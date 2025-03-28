# Fine-gray approach

library(cmprsk)

# event variable: 1 for DCI, 2 for death
# time_to_event : is the time to DCI or death

# Load the data
data_path <- "/Users/jk1/Downloads/temp.csv"
data <- read.csv(data_path)

# for every categorical variable, set data type to categorical
categorical_columns <- c("initial_GCS", "fischer", "wfns", "Coiling", "Clipping",  "location_encoded", "event" )

for (column in categorical_columns) {
  data[[column]] <- as.factor(data[[column]])
}

# Fit the Fine-Gray model
fine_gray_model <- crr(ftime = data$"time_to_event", fstatus = data$event, cov1 = data[, c("initial_GCS", "fischer", "wfns", "Coiling", "Clipping",  "location_encoded")],
                       failcode = 1)

# Summary of the Fine-Gray model
summary(fine_gray_model)
