# Deep Learning in Health 
# Team Project
# Prof. Mitsakakis 
# Zachery Chan, Ibrahim Emam, Jadeyenne Wright

# ----------------- # 
#### LIBRARIES ####
# ----------------- # 
library(keras)
library(dplyr)
library(ggplot2)
library(caret)

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

# Plot class distribution
ggplot(train, aes(x = factor(Sentiment, levels = c(
  "Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"
)))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Sentiment Classes in Training Set",
       x = "Sentiment",
       y = "Count") +
  theme_classic()

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
 
maxlen <- 65 

# getting distribution of tweet length 
ggplot(data.frame(length = word_counts), aes(x = length)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Tweet Lengths (by Word Count)", 
       x = "Number of Words", 
       y = "Count")

# initializing tokenizer to convert strings into tokens 
tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts_train)

# Is a dictionary size of 10K sufficient? Let's explore:

# Extract word frequency counts
word_freqs <- tokenizer$word_counts
freq_df <- data.frame(
  word = names(word_freqs),
  freq = as.numeric(word_freqs)
)

# Sort words by frequency (descending)
freq_df <- freq_df[order(-freq_df$freq), ]
freq_df$rank <- 1:nrow(freq_df)

# Compute cumulative frequency coverage
freq_df$cumulative_freq <- cumsum(freq_df$freq)
freq_df$coverage <- freq_df$cumulative_freq / sum(freq_df$freq)

# Plot cumulative token coverage
ggplot(freq_df[1:20000, ], aes(x = rank, y = coverage)) +
  geom_line(color = "darkgreen") +
  geom_vline(xintercept = 10000, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title = "Cumulative Token Coverage by Vocabulary Size",
    x = "Top N Words",
    y = "Cumulative Coverage"
  ) +
  theme_minimal()

# It seems 10K dictionary size is sufficient

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

# building the embedding matrix (initially all zeros)
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

# -------------------------------- # 
#### CONFUSION MATRIX FUNCTION ####
# -------------------------------- # 

get_confusion_matrix <- function(model, data_test, labels_test_onehot, label_order) {
  # Get predictions as probabilities
  pred_probs <- model %>% predict(data_test)
  
  # Convert one-hot encoded labels and predictions to class indices
  true_classes <- apply(labels_test_onehot, 1, which.max) - 1
  predicted_classes <- apply(pred_probs, 1, which.max) - 1
  
  # Convert to factor with label names
  true_labels <- factor(label_order[true_classes + 1], levels = label_order)
  predicted_labels <- factor(label_order[predicted_classes + 1], levels = label_order)
  
  # Print confusion matrix
  print(confusionMatrix(predicted_labels, true_labels))
}

# ------------------------------- #
##### FUNCTIONS FOR GRAPHING ####
# ------------------------------- #

plotting_loss <- function(x){
  # generating plot for loss 
  loss_plot <- ggplot(all_histories, aes(x = Epoch)) +
    geom_line(aes(y = loss_history, color = "Training Loss"), size = 1) +
    geom_line(aes(y = val_loss_history, color = "Validation Loss"), size = 1, linetype = "dashed") +
    labs(title = paste("Loss Over Epochs for", x), x = "Epoch", y = "Loss", color = "Dataset") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_color_manual(values = c("Training Loss" = "blue", "Validation Loss" = "red"))
  
  return(loss_plot)
}

plotting_accuracy <- function(x){
  # generating plot for loss 
  accuracy_plot <- ggplot(all_histories, aes(x = Epoch)) +
    geom_line(aes(y = accur_history, color = "Training Accuracy"), size = 1) +
    geom_line(aes(y = val_accur_history, color = "Validation Accuracy"), size = 1, linetype = "dashed") +
    labs(title = paste("Accuracy Over Epochs for", x), x = "Epoch", y = "Accuracy", color = "Dataset") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_color_manual(values = c("Training Accuracy" = "purple", "Validation Accuracy" = "orange"))
  
  return(accuracy_plot)
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
  layer_dropout(0.1) %>%
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
history_ffn <- model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

# testing the model
results_ffnn <- model %>% evaluate(data_test, labels_test)
  
results_ffnn

plot(history_ffn)
# 5 epochs sufficient here 

save(history, results_ffnn, file = "results_ffnn.RData")

# loading RData to plot epochs, using this code after running in cloud and downloading saved model 
if (F){
  load("results_model_1.RData")
  
  # getting metrics 
  loss_history <- history_ffn$metrics$loss
  
  accur_history <- history_ffn$metrics$accuracy
  
  val_loss_history <- history_ffn$metrics$val_loss
  
  val_accur_history <- history_ffn$metrics$val_accuracy
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("FFNN, Frozen")
  
  plotting_accuracy("FFNN, Frozen")  
}

# revised FFNN architecture
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading embedding weights and freeze the layer
get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# compiling the revised model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# training the revised model (use all training data without validation)
history_ffn <- model %>% fit(
  data_train, labels_train,
  epochs = 5,
  batch_size = 32
)

# testing the model
results_ffnn <- model %>% evaluate(data_test, labels_test)

results_ffnn

# Get a more detailed idea of classification results
cm_matrix_ffnn <- get_confusion_matrix(model, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_ffnn$table)

# Save to CSV
write.csv(cm_table, "confusion_matrix_ffnn.csv", row.names = FALSE)

# loading RData to plot heatmap, using this code after running in cloud and downloading saved model 
if (F) {
  
  # reading heatmap
  cm_df <- read.csv("confusion_matrix_ffnn.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Locked FFNN",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
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
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load pretrained weights (same as before)
get_layer(model_unfrozen, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # Allow the embedding layer to be trainable

# Compiling the model  
model_unfrozen %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# training the model
history_unfrozen <- model_unfrozen %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 32,
  validation_split = 0.2
)

# testing the model
results_ffnn_unfrozen <- model_unfrozen %>% evaluate(data_test, labels_test)

results_ffnn_unfrozen 

plot(history_unfrozen)
# validation loss is increasing after 5 epochs with no improvement in accuracy
# retrain the model with 5 epochs

save(history_unfrozen,results_ffnn_unfrozen, file = "results_ffnn_unfrozen.RData")

# loading RData to plot epochs, using this code after running in cloud and downloading saved model 
if (F){
  
  load("results_ffnn_unfrozen.RData")
  
  # getting metrics 
  loss_history <- history_unfrozen$metrics$loss
  
  accur_history <- history_unfrozen$metrics$acc
  
  val_loss_history <- history_unfrozen$metrics$val_loss
  
  val_accur_history <- history_unfrozen$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("FFNN, Unfrozen")
  
  plotting_accuracy("FFNN, Unfrozen") 
}

# Rebuild the same model from scratch
model_unfrozen <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load pretrained weights (same as before)
get_layer(model_unfrozen, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # Allow the embedding layer to be trainable

# Compiling the model  
model_unfrozen %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

# training the model
history_unfrozen <- model_unfrozen %>% fit(
  data_train, labels_train,
  epochs = 5,
  batch_size = 32,
)

# testing the model
results_ffnn_unfrozen <- model_unfrozen %>% evaluate(data_test, labels_test)

results_ffnn_unfrozen 

# get a more detailed idea of classification results
cm_matrix_ffnn_unfrozen <- get_confusion_matrix(model_unfrozen, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_ffnn_unfrozen$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_ffnn_unfrozen.csv", row.names = FALSE)

# loading RData to plot heatmap, using this code after running in cloud and downloading saved model 
if (F){
  
  cm_df <- read.csv("confusion_matrix_ffnn_unfrozen.csv")
  
  # Create heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Unlocked FFNN",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# --------------------------------------- # 
#### BUILDING FROZEN RNN + EMBEDDING ####
# --------------------------------------- # 

# building RNN model 
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.05) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading embedding weights and freeze the layer
get_layer(model_rnn, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# compiling the model
model_rnn %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# loading embedding weights and freeze the layer
get_layer(model_rnn, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# training the model
history_rnn <- model_rnn %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 50,
  validation_split = 0.2
)

# testing the model
results_rnn <- model_rnn %>% evaluate(data_test, labels_test)

results_rnn

plot(history_rnn)
# no strong evidence of overfitting at 20 epochs, capping at approx. 15 to mitigate volatility 

save(history_rnn,results_rnn, file = "results_rnn.RData")

# loading RData to plot epochs, using this code after running in cloud and downloading saved model 
if (F){
  
  load("results_rnn.RData")
  
  # getting metrics 
  loss_history <- history_rnn$metrics$loss
  
  accur_history <- history_rnn$metrics$acc
  
  val_loss_history <- history_rnn$metrics$val_loss
  
  val_accur_history <- history_rnn$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("RNN, Single Layer Frozen")
  
  plotting_accuracy("RNN, Single Layer Frozen")
}

# rebuilding RNN model with optimal epochs
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.05) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading embedding weights and freeze the layer
get_layer(model_rnn, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# compiling the model
model_rnn %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# loading embedding weights and freeze the layer
get_layer(model_rnn, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# training the model
history_rnn <- model_rnn %>% fit(
  data_train, labels_train,
  epochs = 12,
  batch_size = 50,
)

# testing the model
results_rnn <- model_rnn %>% evaluate(data_test, labels_test)

results_rnn

# get a more detailed idea of classification results
cm_matrix_rnn <- get_confusion_matrix(model_rnn, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_rnn$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_rnn.csv", row.names = FALSE)

# loading RData to plot heatmap, using this code after running in cloud and downloading saved model 
if (F){
  
  # reading heatmap
  cm_df <- read.csv("confusion_matrix_rnn.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Locked RNN",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


# ----------------------------------------- # 
#### BUILDING RNN (2 LAYER) + EMBEDDING ####
# ----------------------------------------- # 

# building 2 layer RNN model 
model_rnn_2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.2, recurrent_dropout = 0.2,  return_sequences = T) %>%
  layer_simple_rnn(units = 128, dropout = 0.2, recurrent_dropout = 0.2,) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

get_layer(model_rnn_2, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

# compiling model
model_rnn_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer_rmsprop( 
    learning_rate = 0.002),
  metrics = c('accuracy')
)

# training the model
history_rnn_2 <- model_rnn_2 %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 100,
  validation_split = 0.2
)

# testing the model
results_rnn_2 <- model_rnn_2 %>% evaluate(data_test, labels_test)

results_rnn_2 

plot(history_rnn_2)
# dont see any huge differences, capping at 30 

save(history_rnn_2,results_rnn_2, file = "results_rnn_2.RData")

# loading RData to plot model, using this code after running in cloud and downloading saved model 
if (F){
  
  load("/Users/zachery/downloads/results_rnn_2.RData")
  
  # getting metrics 
  loss_history <- history_rnn_2$metrics$loss
  
  accur_history <- history_rnn_2$metrics$acc
  
  val_loss_history <- history_rnn_2$metrics$val_loss
  
  val_accur_history <- history_rnn_2$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("RNN, 2 Layer Frozen")
  
  plotting_accuracy("RNN, 2 Layer Frozen")
}

# get a more detailed idea of classification results
cm_matrix_rnn_2 <- get_confusion_matrix(model_rnn_2, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_rnn_2$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_rnn_2.csv", row.names = FALSE)

# loading RData to plot heatmap, using this code after running in cloud and downloading saved model 
if (F){

  # reading heatmap
  cm_df <- read.csv("confusion_matrix_rnn_2.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Locked, 2-Layer, RNN",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# ---------------------------------------------------- # 
#### BUILDING RNN (2 LAYER) + EMBEDDING, UNFROZEN ####
# ---------------------------------------------------- # 

# building 2 layer RNN model 
model_rnn_3 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.2, recurrent_dropout = 0.2,  return_sequences = T) %>%
  layer_simple_rnn(units = 128, dropout = 0.2, recurrent_dropout = 0.2,) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading pretrained weights 
get_layer(model_rnn_3, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # allowing the embedding layer to be trainable

# compiling model
model_rnn_3 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# training the model
history_rnn_3 <- model_rnn_3 %>% fit(
  data_train, labels_train,
  epochs = 30,
  batch_size = 100,
  validation_split = 0.2
)

# testing the model
results_rnn_3 <- model_rnn_3 %>% evaluate(data_test, labels_test)

results_rnn_3

plot(history_rnn_3)
# no obvious evidence of overfitting at 30 epochs, capping there 

save(history_rnn_3,results_rnn_3, file = "results_rnn_3.RData")

# loading RData to plot model, using this code after running in cloud and downloading saved model 
if (F){
  
  load("results_rnn_3.RData")
  
  # getting metrics 
  loss_history <- history_rnn_3$metrics$loss
  
  accur_history <- history_rnn_3$metrics$acc
  
  val_loss_history <- history_rnn_3$metrics$val_loss
  
  val_accur_history <- history_rnn_3$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("RNN, 2 Layer Unfrozen")
  
  plotting_accuracy("RNN, 2 Layer Unfrozen")
}

# building 2 layer RNN model with optimal epochs
model_rnn_3 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.05, recurrent_dropout = 0.05,  return_sequences = T) %>%
  layer_simple_rnn(units = 128, dropout = 0.05, recurrent_dropout = 0.05,) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.05) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading pretrained weights 
get_layer(model_rnn_3, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # allowing the embedding layer to be trainable

# compiling model
model_rnn_3 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# training the model
history_rnn_3 <- model_rnn_3 %>% fit(
  data_train, labels_train,
  epochs = 30,
  batch_size = 100
)

# testing the model
results_rnn_3 <- model_rnn_3 %>% evaluate(data_test, labels_test)

results_rnn_3

# get a more detailed idea of classification results
cm_matrix_rnn_3 <- get_confusion_matrix(model_rnn_3, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_rnn_3$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_rnn_3.csv", row.names = FALSE)

# loading RData to plot heatmap, using this code after running in cloud and downloading saved model 
if (F){ 
  # reading heatmap
  cm_df <- read.csv("confusion_matrix_rnn_3.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Unlocked, 2-Layer, RNN",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# ---------------------------------------------- # 
#### BUILDING LSTM MODEL + EMBEDDING, FROZEN ####
# ---------------------------------------------- # 

# building lstm model
lstm_model <- keras_model_sequential() %>%
  layer_embedding(input_dim =  max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_lstm(units = 128, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading pretrained weights 
get_layer(lstm_model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()  # allowing the embedding layer to be trainable

lstm_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
  )

lstm_history <- lstm_model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 50,
  validation_split = 0.2
  )

# testing the model
results_lstm <- lstm_model %>% evaluate(data_test, labels_test)

results_lstm 

plot(lstm_history)
# no evidence of loss in accuracy or overfitting at 20 epochs 

save(lstm_history,results_lstm, file = "results_lstm.RData")

# loading RData to plot epochs 
if (F){
  load("results_lstm.RData")
  
  # getting metrics 
  loss_history <- lstm_history$metrics$loss
  
  accur_history <- lstm_history$metrics$acc
  
  val_loss_history <- lstm_history$metrics$val_loss
  
  val_accur_history <- lstm_history$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("LSTM, Frozen")
  
  plotting_accuracy("LSTM, Frozen")
}

# building lstm model with optimal weights 
lstm_model <- keras_model_sequential() %>%
  layer_embedding(input_dim =  max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_lstm(units = 128, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# loading pretrained weights 
get_layer(lstm_model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()  # allowing the embedding layer to be trainable

lstm_model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

lstm_history <- lstm_model %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 50
)

# testing the model
results_lstm <- lstm_model %>% evaluate(data_test, labels_test)

results_lstm 

# get a more detailed idea of classification results
cm_matrix_lstm <- get_confusion_matrix(lstm_model, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_lstm$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_lstm.csv", row.names = FALSE)

# loading RData to plot heatmaps
if (F){
  
  # reading heatmap
  cm_df <- read.csv("confusion_matrix_lstm.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Locked LSTM",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# ------------------------------------------------ # 
#### BUILDING LSTM MODEL + EMBEDDING, UNFROZEN ####
# ------------------------------------------------ # 

# building lstm model
lstm_model_2 <- keras_model_sequential() %>%
  layer_embedding(input_dim =  max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_lstm(units = 128, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load pretrained weights (same as before)
get_layer(lstm_model_2, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # Allow the embedding layer to be trainable

lstm_model_2 %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

lstm_history_2 <- lstm_model_2 %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 50,
  validation_split = 0.2
)

# testing the model
results_lstm_2 <- lstm_model_2 %>% evaluate(data_test, labels_test)

results_lstm_2

plot(lstm_model_2)
# similarly, no obvious evidence of overfitting here 

save(lstm_history_2,results_lstm_2, file = "results_lstm_2.RData")

# loading RData to plot epochs 
if (F){
  
  setwd("/Users/zachery/Downloads/Deep_Learning_Team_Project")
  
  load("results_lstm_2.RData")
  
  # getting metrics 
  loss_history <- lstm_history_2$metrics$loss
  
  accur_history <- lstm_history_2$metrics$acc
  
  val_loss_history <- lstm_history_2$metrics$val_loss
  
  val_accur_history <- lstm_history_2$metrics$val_acc
  
  # # storing as dataframe 
  all_histories <- data.frame(
    Epoch = seq_along(loss_history),
    loss_history,
    accur_history,
    val_loss_history,
    val_accur_history
  )
  
  plotting_loss("LSTM, Unfrozen")
  
  plotting_accuracy("LSTM, Unfrozen")
}

# building lstm model
lstm_model_2 <- keras_model_sequential() %>%
  layer_embedding(input_dim =  max_words,
                  output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_lstm(units = 128, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

# Load pretrained weights (same as before)
get_layer(lstm_model_2, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  unfreeze_weights()  # Allow the embedding layer to be trainable

lstm_model_2 %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

lstm_history_2 <- lstm_model_2 %>% fit(
  data_train, labels_train,
  epochs = 20,
  batch_size = 50
)

# testing the model
results_lstm_2 <- lstm_model_2 %>% evaluate(data_test, labels_test)

results_lstm_2

# get a more detailed idea of classification results
cm_matrix_lstm_2 <- get_confusion_matrix(lstm_model_2, data_test, labels_test, label_order)

# pull confusion matrix
cm_table <- as.data.frame(cm_matrix_lstm_2$table)

# save to CSV
write.csv(cm_table, "confusion_matrix_lstm_2.csv", row.names = FALSE)

# loading RData to plot heatmap
if (F){
  
  # reading heatmap
  cm_df <- read.csv("confusion_matrix_lstm_2.csv")
  
  # creating heatmap
  ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap for Unlocked LSTM",
         x = "True Tweet Sentiment",
         y = "Predicted Tweet Sentiment") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
