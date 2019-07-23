####TO Install Uer Path Env Variables + Run as Administrator#####
#%USERPROFILE%\Anaconda3\Scripts
#%USERPROFILE%\Anaconda3\Library\bin
#%USERPROFILE%\Anaconda3

install.packages("keras")
install.packages("tensorflow")


#cuDNN
# %USERPROFILE%\cuda
# %CUDA_Installation_directory%\bin\cudnn64_7.dll
# % CUDA_Installation_directory %\include\cudnn.h
# % CUDA_Installation_directory %\lib\x64\cudnn.lib
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
# new setup
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64
library(keras)
library(tensorflow)
#GPU
# Works if 
# install_tensorflow(version = "gpu")
# install_keras(tensorflow = "gpu")


#CPU - work. If you install the latest tensorflow_gpu manually. Or switch to base Env for permissions
# use_condaenv("base")
# install_tensorflow()
# install_keras()

# Conda Env Test
# conda create --name tf_gpu tensorflow-gpu 
# activate r_tensorflow
# conda install tensorflow-gpu

####TEst#####
#Check available devices
library(keras)
# #
k<-backend()
sess <-k$get_session()
sess$list_devices()


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
# compile (define loss and optimizer)
model %>% compile(
 loss = 'categorical_crossentropy',
 optimizer = optimizer_rmsprop(),
 metrics = c('accuracy')
)
# train (fit)
model %>% fit(
  x_train, y_train,
  epochs = 5, batch_size = 128,
  validation_split = 0.2
)
model %>% evaluate(x_test, y_test)
model %>% predict_classes(x_test)

