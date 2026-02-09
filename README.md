# Statistical-Consulting
In the framework of a Course, namely Statistical Consulting, at the UCLouvain, we collaborated with two companies : la Société Walonne Des Eaux (SWDE) and Ardennes-Etape (AE). 

Given their datasets, we were expecting to give them a statistical analysis. The course let us free in the technique used to do so.

## SWDE Project
The goal was to identify classes from a raw descriptions. 
The first text preprocessing was apply (cleaning, Tokenization,Stemming, Embedding).
Then, the number of class and the assignation for each sentence was done using Agglomerative Hierarchical Clustering. I simply choose a class based on the minimum distance between the cluster centroid and average sentence position in the embedding space.

Finally, DistilBERT model was fine-tuned on these classes in order to develop a potential verification tool for those descriptions.

The training of the DistilBERT model were not satisfying : difficulty in classifying low frequency classes.
With more time, I would have tried alternative models and thinner cleaning of my dataset.

## AE Project
