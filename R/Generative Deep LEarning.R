#Text Generation####
#Creating the corpus
library(keras)
library(stringr)
#use_condaenv("base")
path <- get_file(
  "nietzsche.txt",
  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
text <- tolower(readChar(path, file.info(path)$size))
cat("Corpus length:", nchar(text), "\n")

maxlen <- 60  # Length of extracted character sequences
step <- 3  # We sample a new sequence every `step` characters

text_indexes <- seq(1, nchar(text) - maxlen, by = step)
# This holds our extracted sequences
sentences <- str_sub(text, text_indexes, text_indexes + maxlen - 1)
# This holds the targets (the follow-up characters)
next_chars <- str_sub(text, text_indexes + maxlen, text_indexes + maxlen)
cat("Number of sequences: ", length(sentences), "\n")
# List of unique characters in the corpus
chars <- unique(sort(strsplit(text, "")[[1]]))
cat("Unique characters:", length(chars), "\n")
# Dictionary mapping unique characters to their index in `chars`
char_indices <- 1:length(chars) 
names(char_indices) <- chars
# Next, one-hot encode the characters into binary arrays.
cat("Vectorization...\n")

x <- array(0L, dim = c(length(sentences), maxlen, length(chars)))
y <- array(0L, dim = c(length(sentences), length(chars)))
for (i in 1:length(sentences)) {
  sentence <- strsplit(sentences[[i]], "")[[1]]
  for (t in 1:length(sentence)) {
    char <- sentence[[t]]
    x[i, t, char_indices[[char]]] <- 1
  }
  next_char <- next_chars[[i]]
  y[i, char_indices[[next_char]]] <- 1
}

#Initiliaize the model again if needed
# model <- keras_model_sequential() %>%
#   layer_lstm(units = 128, input_shape = c(maxlen, length(chars))) %>%
#   layer_dense(units = length(chars), activation = "softmax")
# 
# 
# optimizer <- optimizer_rmsprop(lr = 0.01)
# model %>% compile(
#   loss = "categorical_crossentropy",
#   optimizer = optimizer
# )
#Load the model after the first iteration
model<-load_model_hdf5("Nizhe.h5", custom_objects = NULL, compile = TRUE)

#Sampling the text
sample_next_char <- function(preds, temperature = 1.0) {
  preds <- as.numeric(preds)
  preds <- log(preds) / temperature
  exp_preds <- exp(preds)
  preds <- exp_preds / sum(exp_preds)
  which.max(t(rmultinom(1, 1, preds)))
}

# model %>% fit(x, y, batch_size = 2048, epochs = 1) 
# model %>% fit(x, y, batch_size = 2048, epochs = 10) 
for (epoch in 1:2) { #was 60 initially so far 37
  
  cat("epoch", epoch, "\n")
  
  # Fit the model for 1 epoch on the available training data
  model %>% fit(x, y, batch_size = 1024, epochs = 1)  #was 128, but my video can handle much mroe/2048 seems to be perfect
  
  # Select a text seed at random
  start_index <- sample(1:(nchar(text) - maxlen - 1), 1)  
  seed_text <- str_sub(text, start_index, start_index + maxlen - 1)
  seed_text<-" hard as anal sex view of his beholders. he was neither a ma"
  cat("--- Generating with seed:", seed_text, "\n\n")
  
  for (temperature in c( 0.5, 1.0)
       
       # c(0.2, 0.5, 1.0, 1.2) #Original Values
       
       ) {
    
    cat("------ temperature:", temperature, "\n")
    cat(seed_text, "\n")
    
    generated_text <- seed_text
    
    # We generate 400 characters
    for (i in 1:400) {
      
      sampled <- array(0, dim = c(1, maxlen, length(chars)))
      generated_chars <- strsplit(generated_text, "")[[1]]
      for (t in 1:length(generated_chars)) {
        char <- generated_chars[[t]]
        sampled[1, t, char_indices[[char]]] <- 1
      }
      
      preds <- model %>% predict(sampled, verbose = 0)
      next_index <- sample_next_char(preds[1,], temperature)
      next_char <- chars[[next_index]]
      
      generated_text <- paste0(generated_text, next_char)
      generated_text <- substring(generated_text, 2)
      
      cat(next_char)
    }
    cat("\n\n")
  }
}

save_model_hdf5(model, "Nizhe.h5") #39, 1.2611 loss

#Deep Dreams/Natural Style Transfer - HARD SKIP####
#Repo With sample at 

#Generating Images from latent space####
library(keras)
img_shape <- c(28, 28, 1)
batch_size <- 16
latent_dim <- 2L  # Dimensionality of the latent space: a plane
input_img <- layer_input(shape = img_shape)
x <- input_img %>% 
  layer_conv_2d(filters = 32, kernel_size = 3, padding = "same", 
                activation = "relu") %>% 
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same", 
                activation = "relu", strides = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same", 
                activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same", 
                activation = "relu") 
shape_before_flattening <- k_int_shape(x)
x <- x %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu")
z_mean <- x %>% 
  layer_dense(units = latent_dim)
z_log_var <- x %>% 
  layer_dense(units = latent_dim)

sampling <- function(args) {
  c(z_mean, z_log_var) %<-% args
  epsilon <- k_random_normal(shape = list(k_shape(z_mean)[1], latent_dim),
                             mean = 0, stddev = 1)
  z_mean + k_exp(z_log_var) * epsilon
}
z <- list(z_mean, z_log_var) %>% 
  layer_lambda(sampling)


# This is the input where we will feed `z`.
decoder_input <- layer_input(k_int_shape(z)[-1])
x <- decoder_input %>% 
  # Upsample to the correct number of units
  layer_dense(units = prod(as.integer(shape_before_flattening[-1])),
              activation = "relu") %>% 
  # Reshapes into an image of the same shape as before the last flatten layer
  layer_reshape(target_shape = shape_before_flattening[-1]) %>% 
  # Applies and then reverses the operation to the initial stack of 
  # convolution layers
  layer_conv_2d_transpose(filters = 32, kernel_size = 3, padding = "same",
                          activation = "relu", strides = c(2, 2)) %>%  
  layer_conv_2d(filters = 1, kernel_size = 3, padding = "same",
                activation = "sigmoid")  
# We end up with a feature map of the same size as the original input.
# This is our decoder model.
decoder <- keras_model(decoder_input, x)
# We then apply it to `z` to recover the decoded `z`.
z_decoded <- decoder(z) 

library(R6)
CustomVariationalLayer <- R6Class("CustomVariationalLayer",
                                  
                                  inherit = KerasLayer,
                                  
                                  public = list(
                                    
                                    vae_loss = function(x, z_decoded) {
                                      x <- k_flatten(x)
                                      z_decoded <- k_flatten(z_decoded)
                                      xent_loss <- metric_binary_crossentropy(x, z_decoded)
                                      kl_loss <- -5e-4 * k_mean(
                                        1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), 
                                        axis = -1L
                                      )
                                      k_mean(xent_loss + kl_loss)
                                    },
                                    
                                    call = function(inputs, mask = NULL) {
                                      x <- inputs[[1]]
                                      z_decoded <- inputs[[2]]
                                      loss <- self$vae_loss(x, z_decoded)
                                      self$add_loss(loss, inputs = inputs)
                                      x
                                    }
                                  )
)
layer_variational <- function(object) { 
  create_layer(CustomVariationalLayer, object, list())
} 
# Call the custom layer on the input and the decoded output to obtain
# the final model output
y <- list(input_img, z_decoded) %>% 
  layer_variational() 

vae <- keras_model(input_img, y)
vae %>% compile(
  optimizer = "rmsprop",
  loss = NULL
)

mnist <- dataset_mnist() 
c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist
x_train <- x_train / 255
x_train <- array_reshape(x_train, dim =c(dim(x_train), 1))
x_test <- x_test / 255
x_test <- array_reshape(x_test, dim =c(dim(x_test), 1))
vae %>% fit(
  x = x_train, y = NULL,
  epochs = 10,
  batch_size = batch_size,
  validation_data = list(x_test, NULL)
)


n <- 15            # Number of rows / columns of digits
digit_size <- 28   # Height / width of digits in pixels
# Transforms linearly spaced coordinates on the unit square through the inverse
# CDF (ppf) of the Gaussian to produce values of the latent variables z,
# because the prior of the latent space is Gaussian
grid_x <- qnorm(seq(0.05, 0.95, length.out = n))
grid_y <- qnorm(seq(0.05, 0.95, length.out = n))
op <- par(mfrow = c(n, n), mar = c(0,0,0,0), bg = "black")
for (i in 1:length(grid_x)) {
  yi <- grid_x[[i]]
  for (j in 1:length(grid_y)) {
    xi <- grid_y[[j]]
    z_sample <- matrix(c(xi, yi), nrow = 1, ncol = 2)
    z_sample <- t(replicate(batch_size, z_sample, simplify = "matrix"))
    x_decoded <- decoder %>% predict(z_sample, batch_size = batch_size)
    digit <- array_reshape(x_decoded[1,,,], dim = c(digit_size, digit_size))
    plot(as.raster(digit))
  }
}
par(op)

save_model_hdf5(model, "minstImageGenerator.h5")