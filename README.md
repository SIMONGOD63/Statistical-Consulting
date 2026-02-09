# Statistical-Consulting
In the framework of a Course, namely Statistical Consulting, at the UCLouvain, we collaborated with two companies : la Société Walonne Des Eaux (SWDE) and Ardennes-Etape (AE). 

Given their datasets, we were expecting to give them a statistical analysis. The course let us free in the techniques used to do so and the content of our analysis.
The datasets and reports are not available due to privacy.

## SWDE Project
The goal was to identify classes from a raw descriptions. 
First, text preprocessing was apply (cleaning, Tokenization,Stemming, Embedding).
Then, the number of class and the assignation for each sentence was done using Agglomerative Hierarchical Clustering. I simply choose a class based on the minimum distance between the cluster centroid and average sentence position in the embedding space.

Finally, DistilBERT model was fine-tuned on these classes in order to develop a potential verification tool for those descriptions.

The training of the DistilBERT model were not satisfying : difficulty in classifying low frequency classes.
With more time, I would have tried alternative models and thinner cleaning of my dataset.

At first, a local implementation of mistral was thought about, to discover this new open source technology.
But the size of the model in 2023 and the computation power available made it an impossible direction.

## AE Project
The goal of the project was to identify booking profiles, and more precisely the cases where they were cancelled.
I reduced the dimension using a Variational Auto-Encoder and then KMeans to identify profiles in that dimension.
