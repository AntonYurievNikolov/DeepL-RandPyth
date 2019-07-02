#Initial Tests on MINST####
library(keras)
# model <- keras_model_sequential() %>% 
#   layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
#                 input_shape = c(28, 28, 1)) %>% 
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")
# 
# 
# model <- model %>% 
#   layer_flatten() %>% 
#   layer_dense(units = 64, activation = "relu") %>% 
#   layer_dense(units = 10, activation = "softmax")
# 
# mnist <- dataset_mnist()
# c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
# train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
# train_images <- train_images / 255
# test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
# test_images <- test_images / 255
# train_labels <- to_categorical(train_labels)
# test_labels <- to_categorical(test_labels)
# model %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("accuracy")
# )
# 
# model %>% fit(
#   train_images, train_labels, 
#   epochs = 5, batch_size=64
# )
# 
# results <- model %>% evaluate(test_images, test_labels)
# results

#Working on small datasets####

#Download the data from Kaggle

original_dataset_dir <- "~/Downloads/kaggle_original_data"
base_dir <- "~/Downloads/cats_and_dogs_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)
fnames <- paste0("cat.", 1:1000, ".jpg")
# Do ths 1time for initialization
# file.copy(file.path(original_dataset_dir, fnames), 
#           file.path(train_cats_dir)) 
# fnames <- paste0("cat.", 1001:1500, ".jpg")
# file.copy(file.path(original_dataset_dir, fnames), 
#           file.path(validation_cats_dir))
# fnames <- paste0("cat.", 1501:2000, ".jpg")
# file.copy(file.path(original_dataset_dir, fnames),
#           file.path(test_cats_dir))
# fnames <- paste0("dog.", 1:1000, ".jpg")
# file.copy(file.path(original_dataset_dir, fnames),
#           file.path(train_dogs_dir))
# fnames <- paste0("dog.", 1001:1500, ".jpg")
# file.copy(file.path(original_dataset_dir, fnames),
#           file.path(validation_dogs_dir)) 
# fnames <- paste0("dog.", 1501:2000, ".jpg")
# file.copy(file.path(original_dataset_dir, fnames),
#           file.path(test_dogs_dir))

library(keras)
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)
#Data Preprocessing

train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
  # This is the target directory
  train_dir,
  # This is the data generator
  train_datagen,
  # All images will be resized to 150x150
  target_size = c(150, 150),
  batch_size = 20,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  # This is the target directory
  validation_dir,
  # This is the data generator
  validation_datagen,
  # All images will be resized to 150x150
  target_size = c(150, 150),
  batch_size = 20,
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)

batch <- generator_next(train_generator)
str(batch)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,#should be 30
  validation_data = validation_generator,
  validation_steps = 50
)

#Save the model
model %>% save_model_hdf5("cats_and_dogs_small_1.h5")
plot(history)

#Image generation with Augmentation
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
# Test Augmentations ####
# # We pick one image to "augment"
# fnames <- list.files(train_cats_dir, full.names = TRUE)
# img_path <- fnames[[28]] 
# # Convert it to an array with shape (150, 150, 3)
# img <- image_load(img_path, target_size = c(150, 150))
# img_array <- image_to_array(img)
# img_array <- array_reshape(img_array, c(1, 150, 150, 3))
# # Generated that will flow augmented images
# augmentation_generator <- flow_images_from_data(
#   img_array, 
#   generator = datagen, 
#   batch_size = 1 
# )
# # Plot the first 4 augmented images
# op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
# for (i in 1:4) {
#   batch <- generator_next(augmentation_generator)
#   plot(as.raster(batch[1,,,]))
# }
# par(op)

#New Network with dropout####
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")  

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,#Should be 100
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")
plot(history)
