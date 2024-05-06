# Project Introduction

This project aims to develop machine learning models for Arabic content using advanced techniques such as LSTM, BERT, and BiLSTM. The primary objective is to explore and implement state-of-the-art methods for text classification and analysis in the Arabic language domain.

## Index

1. [Developed Models](#developed-models)
2. [Data](#data)
3. [BERT](#bert)
4. [LSTM](#lstm)
5. [BiLSTM](#bilstm)

## Data

### Overview
The dataset used in this project was collected through web scraping tasks from various student repositories on GitHub. The aim was to compile a comprehensive dataset for further analysis and processing. The data collection process involved gathering information from multiple sources, merging them, and performing necessary preprocessing steps to ensure data consistency and quality.

### Source
The data was primarily sourced from student repositories on GitHub. Each repository contained articles or news articles written in Arabic, covering a variety of topics.

### Data Features
The dataset comprises the following features:

1. **Title**: The title of the article.
2. **Label**: The label assigned to the article, indicating its category (real | fake).Labels may vary in language, punctuation, and synonyms.
3. **Topic**: The topic of the article, which may also vary in language, punctuation, and synonyms for easier merging.
4. **Origine**: The origin of the article, likely referring to the GitHub repository from which it was sourced.
5. **Article_date**: The date of publication of the article(if availble).
6. **Article_content**: The main content of the article(if availble).
7. **Article_correction**: Any corrections of the article content if the article was fake(if availble).

### Preprocessing
Before using the dataset, several preprocessing steps were applied to ensure consistency and ease of usage. This included translating topic names and labels, as well as standardizing punctuation and synonyms across different instances of the dataset. Additionally, some instances required dropping them to make the final data reliable.

# Developed Models   
#

# BERT
## Data Preprocessing

The data preprocessing step involves cleaning and preparing the dataset for training. This includes handling missing values, filtering out short texts, and balancing the class distribution if necessary.

## Model Training

### Tokenization and Dataset Preparation

The Arabic BERT tokenizer is used to tokenize the text data, and the dataset is prepared for training and testing. The texts are tokenized, padded, and converted into PyTorch tensors for input into the BERT model.

### Model Initialization and Optimization

The pre-trained Arabic BERT model for sequence classification is loaded, and an optimizer (e.g., AdamW) is set up for training. Additionally, a learning rate scheduler is configured for adjusting the learning rate during training.

### Training the Model

The model is trained using the training dataset for a specified number of epochs. During each epoch, the model parameters are updated using backpropagation, and the loss is calculated. Training progress, including loss and accuracy, is monitored and printed.

### Evaluating the Model

After training, the model is evaluated using the test dataset. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance in classifying fake and real news articles.

#### Classification Report and Confusion Matrix

A detailed classification report is generated, including precision, recall, F1-score, and support for each class. Additionally, a confusion matrix is computed to visualize the model's performance in classifying instances.

## Saving the Model

Once trained, the BERT or ARABERT model is saved for future use and deployment. The model is serialized using joblib and saved to a specified file path.

                                                                                          
# LSTM

# BiLSTM

## Data Preprocessing

The data preprocessing step involves several tasks such as reading data frames, cleaning text by removing punctuations, links, emojis, HTML tags, and irregular patterns. Additionally, Arabic stop words are removed to improve text quality.

## Making the BILSTM Model

### Initialization of Hyperparameters

The initial step includes setting up hyperparameters required for the model. These parameters include the maximum vocabulary size, tokenizer, and maximum sequence length.

### Defining the LSTM Layers and Dense Layers

The BILSTM (Bidirectional Long Short-Term Memory) model architecture is defined here. The model includes embedding layers, spatial dropout, multiple bidirectional LSTM layers, and dense layers. The purpose of each layer is to process the input text data effectively for classification.

### Compiling the Model

The model is compiled using the sparse categorical cross-entropy loss function and the Adam optimizer. The model is also configured to track accuracy as a metric during training.

### Running the Model with Plots Every Epoch

The BILSTM model is trained on the dataset with plots generated at the end of each epoch to visualize the training and validation loss. This provides insights into the model's performance and helps in monitoring convergence.

### Evaluating the Model

The trained model is evaluated on the test dataset to assess its performance. Accuracy score and other classification metrics such as precision, recall, and F1-score are calculated to measure the model's effectiveness in classifying fake and real news articles.

### Report and Confusion Matrix

A detailed classification report is provided along with a confusion matrix to further analyze the model's performance. The confusion matrix visually represents the model's ability to correctly classify instances and identify any misclassifications.

### Saving the Model in .h5 Format

Once trained, the BILSTM model is saved in .h5 format for future use and deployment in applications requiring fake news detection.

## Plots

Finally, plots showing training and validation accuracy are generated to visualize the model's learning process over epochs.
