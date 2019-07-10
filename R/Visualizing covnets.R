#Visualize intermidate activations####
library(keras)
model <- load_model_hdf5("cats_and_dogs_small_2.h5") #TO DO - add the models to the project
summary(model)  # As a reminder.

img_path <- "~/Downloads/cats_and_dogs_small/test/cats/cat.1673.jpg"
# We preprocess the image into a 4D tensor
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)

img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor <- img_tensor / 255
dim(img_tensor)
plot(as.raster(img_tensor[1,,,]))
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)

#function to plot the channel
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = terrain.colors(12))
}
#visualize the nth channel
n<-7
plot_channel(first_layer_activation[1,,,n])
#Create an image for all channels visualization
dir.create("cat_activations")
image_size <- 58
images_per_row <- 16
for (i in 1:8) {
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row

  png(paste0("cat_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }

  par(op)
  dev.off()
}

#Visualizing the filters#### 
#TO DO :Need to actually check this later - just pasted the code and checked the result so far
model <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE
)
layer_name <- "block3_conv1"
filter_index <- 1
layer_output <- get_layer(model, layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])
 
# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads <- k_gradients(loss, model$input)[[1]] 
 
# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
iterate <- k_function(list(model$input), list(loss, grads))
# Let's test it
c(loss_value, grads_value) %<-%
  iterate(list(array(0, dim = c(1, 150, 150, 3))))
# We start from a gray image with some noise
input_img_data <-
  array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128 
step <- 1  # this is the magnitude of each gradient update
for (i in 1:40) { 
  # Compute the loss value and gradient value
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  # Here we adjust the input image in the direction that maximizes the loss
  input_img_data <- input_img_data + (grads_value * step)
}
deprocess_image <- function(x) {
  
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}

generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # This function returns the loss and grads given the input picture
  iterate <- k_function(list(model$input), list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- 
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img) 
}


library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

library(grid)
library(gridExtra)
dir.create("vgg_filters")
#This melted the CPU - need to try it out with GPU
# for (layer_name in c("block1_conv1", "block2_conv1", 
#                      "block3_conv1", "block4_conv1")) {
#   size <- 140
#   
#   png(paste0("vgg_filters/", layer_name, ".png"),
#       width = 8 * size, height = 8 * size)
#   
#   grobs <- list()
#   for (i in 0:7) {
#     for (j in 0:7) {
#       pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
#       grob <- rasterGrob(pattern, 
#                          width = unit(0.9, "npc"), 
#                          height = unit(0.9, "npc"))
#       grobs[[length(grobs)+1]] <- grob
#     }  
#   }
#   
#   grid.arrange(grobs = grobs, ncol = 8)
#   dev.off()
# }
#Vizualize heatmap of class activation#####
#check this https://arxiv.org/abs/1610.02391.
#To do - Check the sample code, just read so far