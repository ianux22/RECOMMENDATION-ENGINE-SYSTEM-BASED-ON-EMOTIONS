# Recommendation engine system based on emotions

**[Click here to view the PDF presentation with censors](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/Capstone_Prezo.pdf)**

This project looks to develop a recommendation engine system for Frank N Magid & Associates, which identifies several factors to recommend, market, and substitute content to live streaming users of Magid's clients. Magid is a leading consulting and strategy company centred on delivering client-focused professional services geared towards consumer satisfaction. 
The data was generated through a questionnaire completed by respondents after each movie. Questions were asked regarding the sentiments experienced during the movies. In addition, questions were also asked to understand any peculiarities pertaining to how they watched the movies to segment the viewers.  
The project has 3 steps : 
1. Clustering to find new movie genres based on emotions and to segment customers;
2. Develop an XGBoost model to analyze the features that allows a movie to have a high rating;
3. Creation of a recommendation system engine based on emotions.

N.B. Since an NDA was signed, I had to censor or delete some part of code in order to post it here.

## *Phase 1: Feature Engineering and clustering*

***[Code for Correlation, Factor Analysis and Clustering](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/Clustering/01_Clustering_And_New_Metrics.ipynb)***

***[Code for evaluating the clusters created](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/Clustering/03_KM11_Evaluation.ipynb)***

***[Code for modifyng the clusters created](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/Clustering/04_Working_On_KM11.ipynb)***

I'll skip the exploratory data analysis because of NDA and jump directly to **correlations**.
I noticed that sentiments 3 and 17, and 34 and 35 have a extremely high correlation (over 0.85). Since also their meaning is very similar, we decided to merge them in order to avoid multicollinearity problems.
The **sentiments are all binary variables**, so to combine them together we need to sum the columns. This will lead to 3 values :

- 0: the movie is neither sentiment 3 or 17
- 1: the movie is just sentiment 3 or 17
- 2: the movie is both sentiment 3 and 17

I decided to combine together 1 and 2 because just about 10% of the dataset has value "2". Including also the 1s increases the number of elements in the group, limiting the loss of information.

Subsequently, I used the **factor analysis** to create 10 factors in order to analyze the correlation between variables and factors loadings to merge similar sentiments in order to create new variables. The new sentiments created this way are just an average of the combined sentiments.

In the end, I used 3 different clustering algorithms to create new movie genres: Hierarchical clustering, Gaussian mixture models and K-means. The best choice turned up to be a K-mean with 11 groups.

The 11 clusters were evaluated: 
 - observing the correlation between each group and the sentiments;
 - observing the distribution via boxplot of the most correlated sentiments of each group within other groups;
 - performing a TUkey's test to confirm wether the mean is the same between the groups.

The logic is that each cluster should have higher values in some sentiments (ex: horror movies should be more related to "scary" sentiments). To confirm this, we plot the distributions of the most related sentiments of the cluster and use the [Tukey's test](https://www.statisticshowto.com/tukey-test-honest-significant-difference/) to find if means are significantly different from each other.

At this point, we have created 11 clusters. Looking into them, we can find 2 main problems: 
 - The presence of 2 romantic clusters. We solved the issue merging them;
 - The presence of a "dumpster" cluster with all the stuff the algorithm wasn't able to classify. We solved the issue using a random forest model (with about 90% of accuracy) to redistribute those movies in the other clusters.

At this point, We obtained 9 clusters/genres based on emotions rather than genres.

## *Phase 2: XGBoost model to explore what affects rating*

**[Code for Phase 2](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/XGBoost_Model/XGBClassifier%20for%20avg_rating.ipynb)**

In this phase, we used a XGBoost model to explore what are the features that affect most the rating. 
First of all, since the rating range was very short (between 3 and 5), we decided to create a new binary variable: movies with rating 4 or more, are considered "good movies", while the other "bad movies". This way, we can face this problem as a classification task rather than a regression one.
Once created the variable, I divided the data into training, test and validation set, then I tuned the model achieving over 80% of accuracy at the end of the process.
Next, we took advatage of the XGBoost model looking at the feature importance in order to see what are the most important features for a good rating.

## *Phase 3: Creation of a recommendation system engine*

**[Code for phase 3](https://github.com/ianux22/RECOMMENDATION-ENGINE-SYSTEM-BASED-ON-EMOTIONS/blob/master/Recommendation_system_engine/Recommendation%20Engine%20System%20(with%20user%20features).ipynb)**

To create a recommendation system engine we need to compute the similarity among all the movies to create what is called similarity matrix. 
The similarity matrix is a square matrix n x n, where n is the number of movies, in which each column and each row is a movie. It is like the correlation matrix, but instead of having features, this time we have movies, and instead of correlations, the values are similarities.
To compute the similarity score I used 3 metrics: Jaccard index, cosine similarity and euclidean distance. The cosine similarity resulted to be the metric with the best performance.

After the similarity matrix, another matrix is required, which is essentially the result of the cross tabulation between users and movies. If users X has watched movie Y, than the value will be 1, otherwise 0. This helps to filter the recommendations. In this way, a movie the user has already watched would not be recommended to the user again.

Now we're ready to create our recommender. The function ***cosine_recommender*** returns the 10 movies most similar to the input one, then the recommendations are filtered with the user's movie list to eliminate movies already watched by the user.
