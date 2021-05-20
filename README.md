# RECOMMENDATION ENGINE SYSTEM BASED ON EMOTIONS

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

