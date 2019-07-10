#Anomaly Detection with 1d COvnets - data preparation####
library(keras)
library(tibble)
library(readr)
data_dir <- "~/Downloads/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
data <- data.matrix(data[,-1])
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)
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
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
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

#Only Convets#####
# model <- keras_model_sequential() %>% 
#   layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
#                 input_shape = list(NULL, dim(data)[[-1]])) %>% 
#   layer_max_pooling_1d(pool_size = 3) %>% 
#   layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
#   layer_max_pooling_1d(pool_size = 3) %>% 
#   layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
#   layer_global_max_pooling_1d() %>% 
#   layer_dense(units = 1)
# model %>% compile(
#   optimizer = optimizer_rmsprop(),
#   loss = "mae"
# )
# history <- model %>% fit_generator(
#   train_gen,
#   steps_per_epoch = 500, #500
#   epochs = 20, #20
#   validation_data = val_gen,
#   validation_steps = val_steps
# )

#Stacking Gates with 1d Covnets####
model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 50,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
