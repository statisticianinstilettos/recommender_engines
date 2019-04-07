# recommender_engines
A private libary for freelance data science work. These classes can be used to train and deploy user-item recommendations using client provided data. 

![](https://media.giphy.com/media/CIJsP7PsWvZM4/giphy.gif)


Contains recommender classes the following models:

- "Similar items" - Content Filtering via matrix factorization. Takes in tabular features, and/or latent embeddings. 
- "People also purchase" - Collaborative Filtering (solved via matrix decomposition by gradient descent) and Deep Collaborative Filtering.
- "You may also like" - Factorization Machines
- "Pairs well with" - Association rules, Deep outfits. 
- "Describe what you are looking" for bot - Information retrieval (cosine sim with document embeddings. take in any type of document embeddings) (This is the same as content, but it uses sentiment)

Support functions
- various methods for document embeddings (doc2vec, lsa, autoencoders)
