# Project Introduction

This project aims to develop machine learning models for Arabic content using advanced techniques such as LSTM, BERT, and BiLSTM. The primary objective is to explore and implement state-of-the-art methods for text classification and analysis in the Arabic language domain.

## Index

1. [Developed Models](#developed-models)
2. [Data](#data)
3. [BERT](#bert)
4. [LSTM](#lstm)
5. [BLSTM](#blstm)

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

# BERT
                                                                                          
# LSTM

# BLSTM
