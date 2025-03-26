
# Clear the environment - BE CAREFUL!
rm(list = ls())

# Setting the working directory
setwd("/Users/zachery/Downloads/Deep_Learning_Team_Project")

#### Loading packages ####

library(keras)


#### LOADING DATA ####

raw_train <- read.csv("Corona_NLP_test.csv", stringsAsFactors = FALSE, fileEncoding = "latin1")

raw_test <- read.csv("Corona_NLP_test.csv", stringsAsFactors = FALSE, fileEncoding = "latin1")


#### SEtting toeknization parameters ####

max_words <- 10000  # size of the vocabulary
maxlen    <- 65   # maximum length of sequences (i.e. word count)


##### tokenizing training dataset ####

texts_train <- raw_train$OriginalTweet

tokenizer <- text_tokenizer(num_words = max_words) %>%                 
  fit_text_tokenizer(texts_train)


train_sequences <- texts_to_sequences(tokenizer, texts_train)

sequence_lengths <- sapply(train_sequences, length)

hist(sequence_lengths,
     main = "Histogram of Sequence Lengths",
     xlab = "Sequence Length",
     ylab = "Frequency",
     col = "lightblue",
     border = "black")


x_train <- pad_sequences(train_sequences, maxlen = maxlen) # padding here to make them all the same lenght 

##### tokenizing test dataset ####

texts_test  <- raw_test$OriginalTweet

test_sequences  <- texts_to_sequences(tokenizer, texts_test)

x_test <- pad_sequences(test_sequences, maxlen = maxlen) # padding here to make them all the same lenght 


#### Factoring labels ####
label_order = c("Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive") 

train_labels <- factor(raw_train$Sentiment, levels = label_order)

test_labels  <- factor(raw_test$Sentiment, levels = label_order)

# adding -1 here as keras because keras expects indices starting at 0 (python relic)
y_train <- to_categorical(as.integer(train_labels) - 1)

y_test  <- to_categorical(as.integer(test_labels) - 1)

#### Constructing RNN ####

set.seed(123)

#### Building model #####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, 
                  output_dim = 128, 
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1) %>%
  layer_flatten() %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 5, activation = "softmax")

summary(model)

# Compile the model with categorical crossentropy and Adam optimizer
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 100,
  validation_split = 0.2
)

results_model_1 <- model %>% evaluate(x_test, y_test)

results_model_1 
# 1.5137352 0.2996314 

save(history,results_model_1, file = "results_model_1.RData")

if (F){
  load("results_model_1.RData")
  plot(history)
}

#### making model with 2 RNN layers ####

#### Building model #####
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, 
                  output_dim = 128, 
                  input_length = maxlen) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1,  return_sequences = T) %>%
  layer_simple_rnn(units = 128, dropout = 0.1, recurrent_dropout = 0.1,) %>%
  layer_flatten() %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.1) %>%
  layer_dense(units = 5, activation = "softmax")

summary(model)

# Compile the model with categorical crossentropy and Adam optimizer
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 100,
  validation_split = 0.2
)

results_model_2 <- model %>% evaluate(x_test, y_test)

results_model_2 
# 1.5802069 0.2740916 


save(history,results_model_2, file = "results_model_2.RData")

if (F){
  load("results_model_2.RData")
  plot(history)
}
