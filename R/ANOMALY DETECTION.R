# Load the data - just 1 time
# dir.create("~/Downloads/jena_climate", recursive = TRUE)
# download.file(
#   "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
#   "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip"
# )
# unzip(
#   "~/Downloads/jena_climate/jena_climate_2009_2016.csv.zip",
#   exdir = "~/Downloads/jena_climate"
# )

library(tibble)
library(readr)
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
#zoom on daily values
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

#Prepare the Data ####

data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

#Define the Data GEnerator + Train/Validate/TEst Samples generators####
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}


lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size
# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

#Basic Approach for Baseline####
# library(keras)
# model <- keras_model_sequential() %>% 
#   layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
#   layer_dense(units = 32, activation = "relu") %>% 
#   layer_dense(units = 1)
# model %>% compile(
#   optimizer = optimizer_rmsprop(),
#   loss = "mae"
# )
# history <- model %>% fit_generator(
#   train_gen,
#   steps_per_epoch = 500,
#   epochs = 3, #should be 20
#   validation_data = val_gen,
#   validation_steps = val_steps
# )
# plot(history)
#Recurrent Baseline with GRU (gates- this is cheaper)####
library(keras)
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  # Adding this for reguralization with drop out
  # layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
  #           input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 3, #should be 20
  validation_data = val_gen,
  validation_steps = val_steps
)
#Stacking Layers if we are not overfitting####
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 10, #should be 500
  epochs = 2, #should be atleast 40
  validation_data = val_gen,
  validation_steps = val_steps
)
#Bidirectional approach - this is not a good idea for the task I want to test furhter####
# model <- keras_model_sequential() %>% 
#   bidirectional(
#     layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
#   ) %>% 
#   layer_dense(units = 1)
# model %>% compile(
#   optimizer = optimizer_rmsprop(),
#   loss = "mae"
# )
# history <- model %>% fit_generator(
#   train_gen,
#   steps_per_epoch = 500,
#   epochs = 40,
#   validation_data = val_gen,
#   validation_steps = val_steps
# )
