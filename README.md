# AI-Twitter-Customer-Support-Classification

**Twitter Customer Support Classification using Deep Learning**

**Project Overview**

This project focuses on building a deep learning-based text classification system to identify whether a tweet is an inbound customer query or a support response. The project uses real-world Twitter customer support data and follows a complete data science pipeline including data preprocessing, model development, evaluation, and deployment.

Multiple models were implemented and compared, including Logistic Regression, Simple RNN, LSTM, and GRU. Based on performance evaluation, the GRU model was selected as the final model and deployed using Streamlit for real-time prediction.

**Dataset Information**

- Dataset Name: Customer Support on Twitter
- Source: Kaggle
- Link: <https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>
- Columns Used:
  - text: Tweet content
  - inbound: Label indicating whether the tweet is a customer query or a support response

**Data Size Handling and Sampling Strategy**

The original _Customer Support on Twitter_ dataset contains approximately **2.7 million records**, which makes it computationally intensive to process and train deep learning models on a standard personal laptop.

Due to hardware and memory limitations, using the entire dataset was not feasible for model training and experimentation. Therefore, a **data sampling strategy** was adopted.

- **100,000 records** were extracted and used for **model training**
- **25,000 records** were extracted and used for **model validation and performance evaluation**

This approach allows efficient training, faster experimentation, and reliable evaluation while maintaining the real-world characteristics of the dataset. The sampled data size is sufficient to capture language patterns and achieve meaningful performance using deep learning models.

**Problem Statement**

To classify Twitter messages as inbound customer support queries or outbound support responses using natural language processing and deep learning techniques.

**Technologies Used**

- Python
- Pandas, NumPy
- Natural Language Processing (NLP)
- Scikit-learn
- TensorFlow / Keras
- Deep Learning (RNN, LSTM, GRU)
- Streamlit

**Methodology**

- Data extraction and selection of relevant columns
- Text preprocessing including cleaning, tokenization, and padding
- Model training using Logistic Regression, RNN, LSTM, and GRU
- Performance comparison using classification metrics
- Selection of the best-performing GRU model
- Deployment of the trained model using Streamlit

**Models Implemented**

- Logistic Regression (baseline model)
- Simple RNN
- LSTM
- GRU (final selected model)

**Model Evaluation**

Models were evaluated using accuracy and classification metrics. The GRU model demonstrated better performance and generalization compared to other models and was therefore selected for deployment.

**Deployment**

The trained GRU model was deployed using Streamlit, allowing users to input tweet text and receive real-time predictions along with confidence scores.
