
# Project: **Medicine Review Classification**

### Team: **TEAM0-03: THUNDER**
**Team Members**:
- Mihir Pandya
- Aayan
- Shashwat Sharma (Leader)
- Manish

---

## **Problem Statement**
The objective of this project is to develop a predictive model to classify medicines into one of three categories — **optimal**, **semi-optimal**, or **poor** — based on user reviews. The goal is to analyze the text of medicine reviews and predict the quality class of each medicine.

---

## **Dataset Description**
The dataset consists of patient reviews on specific drugs, their related conditions, and a patient rating reflecting overall satisfaction. This data was obtained from online pharmaceutical review sites to study sentiment analysis over multiple facets of the drug experience.

Key features in the dataset:
1. **drugName**: Name of the drug.
2. **condition**: The medical condition treated by the drug.
3. **review**: Text reviews provided by patients.
4. **rating**: Numerical rating (1-10) reflecting satisfaction.
5. **date**: Date the review was posted.
6. **usefulCount**: Number of users who found the review helpful.

---

## **Libraries Used**
1. **Data Manipulation**:
   - **Pandas**: For handling tabular data.
   - **Numpy**: For numerical computing.

2. **Natural Language Processing (NLP)**:
   - **nltk** (Natural Language Toolkit): Used for tokenization, stemming, and lemmatization.
   - **re** (Regular Expressions): For text pattern matching and manipulation.
   - **WordNetLemmatizer**: For reducing words to their base form.

3. **Text Feature Extraction**:
   - **CountVectorizer**: To convert text documents into a matrix of token counts.
   - **TF-IDF Vectorizer**: To compute the Term Frequency-Inverse Document Frequency score for feature extraction.

4. **Machine Learning**:
   - **Logistic Regression**: For classification.
   - **Random Forest Classifier**: For feature importance and classification.
   - **SVC**: For classification

---

## **Methodology**
1. **Text Preprocessing**: 
   - Removing punctuation, stopwords, and special characters.
   - Tokenization, lemmatization, and converting text to lowercase.
   
2. **Feature Extraction**:
   - **Bag of Words** and **TF-IDF** techniques used to convert text data into numerical features.
   
3. **Modeling**:
   - Logistic Regression, Random Forest, and SVC were applied.
   - **GridSearchCV** was used to perform hyperparameter tuning for the best model configuration.
   
4. **Evaluation**:
   - The models were evaluated based on accuracy, and the best-performing model was selected for final predictions.

---

## **Future Improvements**
- Further tuning of hyperparameters for the classifiers.
- Using more complex models, Ensemble methods like XGBoost.
- Experimenting with more advanced NLP techniques like **word embeddings** (e.g., Word2Vec or GloVe).
- Incorporating sentiment analysis to enhance the model’s understanding of review content.

---