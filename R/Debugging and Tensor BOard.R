####A simple test with MINST#####
#Check available devices
library(keras)
# # use_condaenv("r-tensorflow")
# k<-backend()
# sess <-k$get_session()
# sess$list_devices()


# input layer: use MNIST images
mnist <- dataset_mnist()
x_train <- mnist$train$x; y_train <- mnist$train$y
x_test <- mnist$test$x; y_test <- mnist$test$y
# reshape and rescale
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255; x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# defining the model and layers
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu',
              input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


#Define the CallBacks####

#setup the tensorboard
# dir.create("logs")
tensorboard("logs")
callbackList<- list(
  callback_early_stopping(
    monitor = "acc",
    patience = 0 #how many Epochs we will wait for improvement
  ),
  callback_model_checkpoint(
    filepath = "BestModelSoFar.h5",
    monitor = "categorical_crossentropy",
    save_best_only = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 2
  ),
 #Tensorbord ####
 callback_tensorboard(
   log_dir = "logs",
   histogram_freq = 1,
   embeddings_freq = 1
 )

)
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

model %>% fit(
  x_train, y_train,
  epochs = 5, batch_size = 128,
  validation_split = 0.2,
  callbacks = callbackList ,
  validation_data = list(x_test,y_test)#need to ad this for the callbcks
)
# model %>% evaluate(x_test, y_test)
# model %>% predict_classes(x_test)

