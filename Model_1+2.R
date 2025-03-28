# Deep Learning in Health 
# Team Project
# Prof. Mitsakakis 
# Zachery Chan, Ibrahim Emam, Jadeyenne Wright

# ----------------- # 
##### LIBRARIES ####
# ----------------- # 
library(keras)
library(dplyr)
library(ggplot2)

# ------------------------ # 
#### LOADING DATASETS ####
# ------------------------ # 

# clearing the environment
rm(list = ls())

# Setting the working directory (note, cannot use this line in the cloud environment)
setwd("/Users/zachery/Downloads/Deep_Learning_Team_Project")

# Load in the datasets
train <- read.csv("Corona_NLP_train.csv")
test <- read.csv("Corona_NLP_test.csv")

# --------------------------- # 
#### PRE-PROCESSING & EDA ####
# --------------------------- # 

dim(train)

dim(test)

# NEED MORE FOR EDA?

# pre - processing 

# Keep only the columns that are needed (i.e. tweets + sentiment)
train <- train[,c(5,6)]
test <- test[,c(5,6)]

# setting seed 
set.seed(123)

# shuffling data 
train <- train[sample(nrow(train)), ]

# ------------------------------- # 
#### TOKENIZING & EMBEDDING ####
# ------------------------------- # 

# getting training tweets and labels
texts_train <- train$OriginalTweet

labels_train <- train$Sentiment

texts_test <- test$OriginalTweet

labels_test <- test$Sentiment

# factoring test labels

# setting extremely negative as 0, extremely positive as 4 
label_order = c("Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive") 

labels_train <- as.integer(factor(train$Sentiment, levels = label_order)) -1 

labels_test <- as.integer(factor(test$Sentiment, levels = label_order)) -1 

num_classes <- length(unique(labels_test))

num_classes 
# 5 classes which is correct 

# tokenizing the text + setting tokenizer parameters
max_words <- 10000

# determining optimal number of features (reducing padding)
# need to get rid of any unusual characters (e.g. emojis) for tokenizer to work
texts_train <- iconv(texts_train, from = "UTF-8", to = "ASCII", sub = "")

texts_test <- iconv(texts_test, from = "UTF-8", to = "ASCII", sub = "")

# visualizing distribution of tweet lengths by number of words
word_counts <- sapply(strsplit(texts_train, "\\s+"), length)

max(word_counts)
# will set max feature = 65
 
maxlen = 65 

# getting distribution of tweet length 
ggplot(data.frame(length = word_counts), aes(x = length)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Tweet Lengths (by Word Count)", 
       x = "Number of Words", 
       y = "Count")

# initializing tokenizer to convert strings into tokens 
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts_train)

# converting tweets into tokens (words)
sequences_train <- texts_to_sequences(tokenizer, texts_train)

sequences_test <- texts_to_sequences(tokenizer, texts_test)

# padding sequences for equal length 
data_train <- pad_sequences(sequences_train, maxlen = maxlen)

data_test <- pad_sequences(sequences_test, maxlen = maxlen)

# one-hot encoding the labels
labels_train <- to_categorical(labels_train, num_classes = num_classes)

labels_test <- to_categorical(labels_test, num_classes = num_classes)

# loading the GloVe embeddings
embedding_dim <- 100 # why using 100 embedding dimensions here? 
lines <- readLines("glove.6B.100d.txt")

# creating new environment where we get the numeric vectors for embedding using GloVe
embeddings_index <- new.env(hash = TRUE)
for (line in lines) {
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  coefs <- as.numeric(values[-1])
  embeddings_index[[word]] <- coefs
}

# building the embedding matrix (initally all zeros)
word_index <- tokenizer$word_index
embedding_matrix <- matrix(0, nrow = max_words, ncol = embedding_dim)

# filling the embedding matrix only if the word is present
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      embedding_matrix[index + 1, ] <- embedding_vector
    }
  }
}

# ----------------------------------- # 
#### BUILDING FFNN + 1 EMBEDDING ####
# ----------------------------------- # 

# FFNN architecture
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading embedding weights and freeze the layer
get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# compiling the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# training the model
history <- model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 100,
  validation_split = 0.2
)

# testing the model
results_ffnn <- model %>% evaluate(data_test, labels_test)
  
results_ffnn

save(history, results_ffnn, file = "results_ffnn.RData")

if (F){
  load("results_model_1.RData")
  plot(history)
}

# ------------------------------------------- # 
#### BUILDING UNFROZEN FFNN + 1 EMBEDDING ####
# ------------------------------------------- # 

# Rebuild the same model from scratch
model_unfrozen <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load pretrained weights (same as before)
get_layer(model_unfrozen, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # Allow the embedding layer to be trainable

# Compile and train the model
model_unfrozen %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history_unfrozen <- model_unfrozen %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 100,
  validation_split = 0.2
)

plot(history_unfrozen)
# There seems to be improvement after unfreezing the embedding layer

# testing the model
results_ffnn_unfrozen <- model %>% evaluate(data_test, labels_test)

results_ffnn_unfrozen 

save(history_unfrozen,results_ffnn_unfrozen, file = "results_ffnn_unfrozen.RData")

if (F){
  load("results_ffnn_unfrozen.RData")
  plot(history_unfrozen)
}

# -------------------------------- # 
#### BUILDING RNN + EMBEDDING ####
# -------------------------------- # 

# building RNN model 
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = num_classes, activation = "softmax")

# compiling the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = c('accuracy')
)

# training the model
history_rnn <- model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 100,
  validation_split = 0.2
)

# testing the model
results_rnn <- model %>% evaluate(data_test, labels_test)

results_rnn

save(history_rnn,results_rnn, file = "results_rnn.RData")

if (F){
  load("results_rnn.RData")
  plot(history_rnn)
}

# ----------------------------------------- # 
#### BUILDING RNN (2 LAYER) + EMBEDDING ####
# ----------------------------------------- # 

# building 2 layer RNN model 
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1,  return_sequences = T) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1,) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = num_classes, activation = "softmax")

# compiling model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = c('accuracy')
)

# training the model
history_rnn_2 <- model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 100,
  validation_split = 0.2
)

# testing the model
results_rnn_2 <- model %>% evaluate(data_test, labels_test)

results_rnn_2 

save(history_rnn_2,results_rnn_2, file = "results_rnn_2.RData")

if (F){
  load("results_rnn_2.RData")
  plot(history)
}
