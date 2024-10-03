# compare Python differencing and de-differencing in 'TimeSeriesDifferencing'
#   against R's 'diff' and 'diffinv' functions

input_filepath <- paste(
  getwd(), "output", "model01", "differencing05",
  "original_and_dediff.csv",
  sep = "/"
)

df <- read.csv(input_filepath)

head(df)

train_diff <- diff(df$ts_train, lag = 6, differences = 1)
all(train_diff == df$ts_train_season_diff_1_padded[7:dim(df)[1]])
# differencing between Python and R match
# > all(train_diff == df$ts_train_season_diff_1_padded[7:dim(df)[1]])
# [1] TRUE


diff_dediff_1 <- diffinv(
  train_diff,
  lag = 6, differences = 1, xi = df$ts_train[1:6]
)
all((diff_dediff_1 - df$ts_train) < 1e-10)
# de-differencing in R recovers original time series
# > all((diff_dediff_1 - df$ts_train) < 1e-10)
# [1] TRUE


fittedvalues_1 <- df$fittedvalues_1[7:dim(df)[1]]
length(train_diff) == length(fittedvalues_1)
# verify correct amount of pre-pended zero-padding has been removed
# > length(train_diff) == length(fittedvalues_1)
# [1] TRUE
# > length(train_diff) == length(df$fittedvalues_1[7:dim(df)[1]])
# [1] TRUE


fitted_dediff_1 <- diffinv(
  fittedvalues_1,
  lag = 6, differences = 1, xi = df$ts_train[1:6]
)
length(fitted_dediff_1)

fitted_dediff_1 <- diffinv(
  df$ts_train_season_diff_1_padded[7:dim(df)[1]] + fittedvalues_1,
  lag = 6, differences = 1, xi = df$ts_train[1:6]
)
length(fitted_dediff_1)
all(fitted_dediff_1 == df$fittedvalues_dediff_1)
# de-differencing fitted values provides same answer in R and Python
# > all(fitted_dediff_1 == df$fittedvalues_dediff_1)
# [1] TRUE


x <- 1:dim(df)[1]
plot(x, df$ts_train, type = "l", col = "black", lwd = 2, xlab = "Time", ylab = "Value")
lines(x, df$fittedvalues_dediff_1, col = "blue", lwd = 2)
lines(x, fitted_dediff_1, col = "green", lwd = 2)
