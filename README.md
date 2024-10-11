# The Flick Pick Engine

## Table of Contents
1. [Intelligent Movie Recommendations with Neural Networks and NLP](#intelligent-movie-recommendations-with-neural-networks-and-nlp)
2. [Author](#author-dsfpt07-group-12)
3. [Overview](#overview)
4. [Problem Statement](#problem-statement)
5. [Data Understanding](#data-understanding) 
   - [Data Sources](#data-sources)
   - [Key Steps in Data Understanding](*key-steps-in-data-understanding)
6. [Data Preparation](#data-preparation)
7. [Modelling](#modelling)
   - [Model Evaluation](#model-evaluation)
8. [Model Tuning](#model-tuning)
   - [Tuned Model Evaluation](#tuned-model-evaluation)
9. [Conclusion](#conclusion)
10. [Recommendations](#recommendations)
11. [Possible next steps](#possible-next-steps)
12. [For more information](#for-more-information)
13. [Repository Structure](#repository-structure)

# Intelligent Movie Recommendations with Neural Networks and NLP
![alt text](image-4.png)

### Author: DSFPT07 Group 12
- Branely Ope
- Brian Kipngetich
- Cynthia Atieno
- Geoffrey Mwangi
- Linet Maz'susa
- Maureen Wanjeri
- Mercy Silali

## Overview

## Problem Statement
![image](https://github.com/user-attachments/assets/6359c57a-c196-4b2c-a723-ec73032a196d)

In today's digital age, the sheer volume of available movies has grown exponentially, leading to an overwhelming choice paralysis for viewers seeking content that aligns with their personal tastes. Traditional recommendation systems often fall short by providing generic suggestions based on popularity or simplistic user behaviors, failing to capture the nuanced preferences of individual users. This lack of personalization results in a suboptimal viewing experience, where users spend more time searching for movies than enjoying them.

The FlickPick Engine aims to solve this problem by developing an intelligent movie recommendation system that delivers highly personalized and relevant suggestions to users. By leveraging advanced machine learning techniques—specifically neural networks for collaborative filtering and Natural Language Processing (NLP) for content analysis—the system can understand and interpret both user preferences and movie attributes on a deeper level.

By addressing the challenges of information overload and impersonal recommendations, the FlickPick Engine enhances the movie discovery process. It empowers users to effortlessly find films that resonate with their unique tastes, thereby improving user satisfaction and engagement with the platform.
## Data Understanding
To develop the FlickPick Engine, we utilized the MovieLens 20M Dataset, a widely recognized dataset in the recommendation systems domain. This dataset provides a rich source of user ratings, movie metadata, and user-generated tags, enabling the creation of a robust and personalized movie recommendation system.

### Data Sources
1. Ratings Data ([ratings.csv](https://github.com/geomwangi007/Movie-Recommendation-System-Group_12-Project/blob/main/Data/ml-latest-small/links.csv)): Contains 25 million ratings ranging from 0.5 to 5.0, provided by 162,541 users on 62,423 movies.
2. Movies Data ([movies.csv](https://github.com/geomwangi007/Movie-Recommendation-System-Group_12-Project/blob/main/Data/ml-latest-small/movies.csv)): Includes movie IDs, titles, and genres for all movies rated in the dataset.
3. Tags Data ([tags.csv](https://github.com/geomwangi007/Movie-Recommendation-System-Group_12-Project/blob/main/Data/ml-latest-small/tags.csv)): Consists of 1.1 million user-generated tags applied to movies, offering additional contextual information.
4. Links Data ([links.csv](https://github.com/geomwangi007/Movie-Recommendation-System-Group_12-Project/blob/main/Data/ml-latest-small/links.csv)): Provides identifiers that link MovieLens movie IDs with IDs from other databases like IMDb and TMDb.
   
**Key Steps in Data Understanding**

- Initial Exploration: Reviewed the datasets to understand the structure, key variables, and relationships between the datasets.
- Data Cleaning: Addressed missing values and data inconsistencies, and merged the datasets to create a unified dataset for modeling.
- Feature Selection: Identified key features that are most likely to impact the primary contributory cause of accidents.

**Data Visualization**

![alt text](image.png)
- Skew Towards Higher Ratings: The distribution is right-skewed, with most ratings clustering around 3, 4, and 5. Ratings of 4 are the most frequent, indicating that users generally rate movies quite positively.

- Few Low Ratings: There are relatively few ratings of 1 and 2, which suggests that users may be less likely to give movies extremely low scores, or that most of the movies in the dataset are well-regarded.

- Peak at Rating 4: The highest count of ratings is around 4, suggesting that many users find the movies to be above average but not necessarily perfect.
  

This type of distribution is common in user-driven ratings, where users tend to be more inclined to rate items positively than negatively.
![alt text](image-1.png)

- The distribution is often long-tailed, with a few users providing many ratings and many users providing few ratings.
- Power-Law Distribution: The distribution exhibits a power-law trend, where most users rate very few movies, and only a few users provide a large number of ratings. This type of behavior is typical in user-generated content datasets, often referred to as the "long tail."

- Majority Have Rated Few Movies: A significant number of users have rated fewer than 100 movies, which suggests that casual users dominate the dataset.

- Heavy Users: There are a small number of users who have rated over 500 movies, with some even rating over 1000. These "heavy users" contribute disproportionately to the number of total ratings in the dataset.
Genre Popularity

![alt text](image-2.png)


- Identify dominant genres (e.g., Drama, Comedy).
  
- Helps in balancing genre representation in recommendations.

![alt text](image-3.png)

Each cell indicates a user's rating for a specific movie, with colors ranging from dark (lower ratings) to bright yellow (higher ratings). Here are some insights:

1. Sparse Matrix: The heatmap is quite sparse, indicating that most users rate only a small subset of available movies. This pattern is common in user-item rating matrices for movie recommendation systems.

2. Ratings Distribution:

- Ratings are distributed across a range of values from around 0.5 to 5.
- The color gradient represents different ratings, with dark blue for low ratings (close to 0.5) and bright yellow for high ratings (close to 5).
3. Few Highly Rated Items: There are only a few instances of bright yellow cells, implying that while there are high ratings, they are not very common. Users tend to rate movies more conservatively or moderately.

4. Clusters of Activity: There are small clusters of ratings, which might indicate a group of popular movies that have been rated by multiple users. This suggests that some movies have broader appeal while many others have only a handful of ratings.

5. Recommendation System Implication: The sparsity in the matrix is a common challenge in recommendation systems, making collaborative filtering techniques ideal since they can leverage the similarities between users or items to fill in the missing ratings.

## Data Preparation
## Modelling
### Model Evaluation
## Model Tuning
### Tuned Model Evaluation
## Conclusion
- Limitations
## Recommendations
## Possible next steps
## For more information
Detailed documentation of the data, methodologies, and code used in this project is available upon request.

## Repository Structure
```bash
Project-Ph4-The-Flick-Pick-Engine-main/
├── notebooks/
│   └── main_index.ipynb       # Main Jupyter Notebook with code and analysis
├── readme_images/             # Images used in the README.md
├── visuals/                   # Visualizations generated during the analysis
├── Non_technical_Presentation/ # Folder containing slides or presentations for a non-technical audience
├── .gitignore                 # Git ignore file
└── README.md                  #

