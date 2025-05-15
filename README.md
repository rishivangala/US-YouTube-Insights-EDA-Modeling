# US YouTube Insights EDA Modeling

### Overview

This project analyzes and models the characteristics of trending YouTube videos in the United States. Using a dataset of U.S. trending videos, we explore engagement patterns, identify optimal posting strategies, and build machine learning models to predict video performance categories and content type (category).

### Objectives

1. Explore key attributes of trending videos (e.g., categories, engagement metrics, posting times).

2. Understand how features like title length, posting hour, and category impact engagement.

3. Predict performance class (Low, Medium, High) based on metadata (no leakage).

4. Predict a video's content category using only its title and metadata.

### Dataset

Source: Kaggle - https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset/data

Files Used: US_youtube_trending_data.csv, US_category_id.json

Size: ~270,000 trending video records

### Tech Stack

* Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Statsmodels)

* NLP tools: CountVectorizer, TfidfVectorizer (optional)

* Jupyter Notebook for analysis

### Exploratory Analysis

Subcategories Explored:

* Video Category Impact (engagement & frequency by category)

* Engagement Ratios (likes/views, comments/views)

* Posting Time Impact (by hour and day of week)

* Title Characteristics (word count, punctuation, clickbait signals)

### Models Built

#### 1. Performance Prediction (Low / Medium / High)

  * Target: Engagement-classified performance labels

  * Models: Logistic Regression, Random Forest

  * Features: Posting hour, day of week, title word count, exclamations, category (encoded)

#### 2. Category Prediction

  * Target: category_name

  * Model: Multinomial Naive Bayes (text only)

  * Features: Title text vectorized using CountVectorizer

### Key Results

* Engagement peaks in Music and Gaming categories.
  
* Highest Average Engagement Score seen in Nonprofits & Activism and Comedy

* Videos posted between 6–10 AM on Monday and Tuesday gain higher average views.

* Random Forest achieved 81% test accuracy on performance classification.
  
* Logistic Regression achieved 83% test accuracy on performance classification.

* Naive Bayes could predict video category from titles with fair accuracy (~55%+).

### Next Steps

* Add more advanced NLP (n-grams, sentiment analysis)

* Fine-tune models using GridSearchCV

* Explore video title generation ideas based on top-performing patterns

### Author

[Rishi Vangala][https://www.linkedin.com/in/RishiVangala/]

### License

Open-source under the MIT License

