# Deep Learning in Health - Team Project
# Ibrahim's Section - Simple FFNN with Embedding Layer
# March 23, 2025

# Clear the environment
rm(list = ls())

# Setting the working directory (note, cannot use this line in the cloud environment)
#setwd("C:/Users/ibrah/Desktop/Deep Learning in Health/Team Project")

# Load packages
library(keras)
library(dplyr)
library(ggplot2)

# Load in the datasets
train <- read.csv("Corona_NLP_train.csv")
test <- read.csv("Corona_NLP_test.csv")

# Keep only the columns that are needed
train <- train[,c(5,6)]
test <- test[,c(5,6)]

# Shuffle the order of the data to avoid selection bias with the validation set
set.seed(123)
train <- train[sample(nrow(train)), ]

### Build a simple NN with an embedding layer ###


texts <- train$OriginalTweet
labels <- train$Sentiment

# Convert labels to integers starting from 0
labels <- as.integer(as.factor(labels)) - 1
num_classes <- length(unique(labels))

# Tokenize the text
max_words <- 10000
maxlen <- 65

# Need to get rid of any unusual characters (e.g. emojis) for tokenizer to work
texts <- iconv(texts, from = "UTF-8", to = "ASCII", sub = "")

# Visualize distribution of tweet lengths by number of words
word_counts <- sapply(strsplit(texts, "\\s+"), length)
max(word_counts)
ggplot(data.frame(length = word_counts), aes(x = length)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Tweet Lengths (by Word Count)", 
       x = "Number of Words", 
       y = "Count")

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)
data <- pad_sequences(sequences, maxlen = maxlen)

# One-hot encode the labels
labels <- to_categorical(labels, num_classes = num_classes)

# Load GloVe embeddings
embedding_dim <- 100
lines <- readLines("glove.6B.100d.txt")

embeddings_index <- new.env(hash = TRUE)
for (line in lines) {
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  coefs <- as.numeric(values[-1])
  embeddings_index[[word]] <- coefs
}

# Build the embedding matrix (initally all zeros)
word_index <- tokenizer$word_index
embedding_matrix <- matrix(0, nrow = max_words, ncol = embedding_dim)

# Fill the embedding_matrix only if the word is present
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      embedding_matrix[index + 1, ] <- embedding_vector
    }
  }
}

# Build the model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load embedding weights and freeze the layer
get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# Compile and train the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  data, labels,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history)

### Try unfreezing the weights of the embeddings ###

# Rebuild the same model from scratch
model_unfrozen <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.5) %>%
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
  data, labels,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

plot(history_unfrozen)

# There seems to be improvement after unfreezing the embedding layer