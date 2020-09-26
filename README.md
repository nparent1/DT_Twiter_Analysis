# DT_Twiter_Analysis

This personal project looks at data from president Trump's Twitter. Donald Trump used to tweet from an iphone while his campaign staff tweeted through his account on Android, so this data can be used to train a neural network to predict who wrote his newer tweets based on a set of features, to include how often he used certain words and how long the tweets were.

I used both logistic regression and a simple neural network, both implemented in matlab (some of this code is from the Stanford ML Coursera). They ended up working about the same.

The full data set is too large to upload here but here's a link: https://drive.google.com/file/d/1NF8q98BrK2QK5dcGuxY15O4JI-Xm6dhW/view?usp=sharing

Best to look at Final_Solution.pdf


This is still a work in progress. Goal is to now take the trained model and create some friendly ui to input his newer tweets (which no longer have this clear characterization) and predict who wrote the tweet.

# What I learned

It was super rewarding to implement the foundational processes that make up machine learning, from the simpler logistic regression with its gradient descent to a neural network (forward propagating to make prediction then back propagating to adjust weights). This was what I spent most of the Machine Learning Coursera learning.

It was also especially cool to see the power of machine learning applied to a very fun problem, predicting the author of a famous person's tweets. The data was rich with features and very easy to grasp the meaning behind.

# Where to go from here

The goal would be to make some sort of interface which could then take in a tweet and extract out the features we currently use, then feed it into the model and see what it predicts.

Also, the model currently only gets ~86% accuracy on the test set. This is pretty good but it has me wondering ways to improve the model. I played around with parameters a bit but it doesn't seem like that's the issue. I think perhaps the predictive power of the features currently being extracted (mostly just a count of common words used) may not be too strong. But I think a tweet is rich with more information that could hopefully be extracted to serve as a better feature.

# Interesting Observations

The logistic regression actually performed about as well as the neural network. I think in a world ridden with buzz words like "neural network," it was really cool to see just how powerful logistic regression could be. As a student of mathematics I really appreciate the elagance of age-old established statistical inference making.
