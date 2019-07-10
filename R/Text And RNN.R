#####One-Hot Encoding with Keras, will skip the manual samples#####
library(keras)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# Creates a tokenizer, configured to only take into account the 1,000 
# most common words, then builds the word index.
tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)
# Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, samples)
# You could also directly get the one-hot binary representations. Vectorization 
# modes other than one-hot encoding are supported by this tokenizer.
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
# How you can recover the word index that was computed
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

#Word VEctors/Embedding ####

# Number of words to consider as features
max_features <- 10000
maxlen <- 20
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

model <- keras_model_sequential() %>% 
  # We specify the maximum input length to our Embedding layer
  # so we can later flatten the embedded inputs
  layer_embedding(input_dim = 10000, output_dim = 8, 
                  input_length = maxlen) %>% 
  # We flatten the 3D tensor of embeddings 
  # into a 2D tensor of shape `(samples, maxlen * 8)`
  layer_flatten() %>% 
  # We add the classifier on top
  layer_dense(units = 1, activation = "sigmoid") 
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

#Pre-Trained Word Embedding + Tests####


imdb_dir <- "~/Downloads/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

#TOkenize the data
library(keras)
maxlen <- 100                 # We will cut reviews after 100 words
training_samples <- 200       # We will be training on 200 samples
validation_samples <- 1000    # We will be validating on 10000 samples
max_words <- 10000            # We will only consider the top 10,000 words in the dataset
tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")

#Split the data and Randomize the index
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1): 
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

#Pre-Process the embedding
glove_dir = '~/Downloads/glove.6B'
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

#Build the Embedding Matrix
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      # Words not found in the embedding index will be all zeros.
      embedding_matrix[index+1,] <- embedding_vector
  }
}
#Defining the Model we will use
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

#Load the embedding and freeze
get_layer(model, index = 1) %>% 
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights()

#Run and Evaluate
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")

#Test the data vs the model
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)

model %>% 
  load_model_weights_hdf5("pre_trained_glove_model.h5") %>% 
  evaluate(x_test, y_test, verbose = 0)