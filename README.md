# recommender_engines
A private libary for freelance data science work. These classes can be used to train and deploy recommendations using client provided data. 

![](https://media.giphy.com/media/CIJsP7PsWvZM4/giphy.gif)


Contains recommender classes the following models:

- Content Filtering via matrix factorization. Takes in tabular features, and/or latent embeddings. 
- User-based and Item-based Collaborative Filtering
  * Matrix decomposition by gradient descent
  * Nonnegative matrix factroization
  * Probabalistic Matrix Factorization
  * Deep Collaborative Filtering with autoencoders
- Factorization Machines
- Association rules
- Deep outfits. ("pairs well with")
- "Describe what you are looking" for bot - Information retrieval (cosine sim with document embeddings. take in any type of document embeddings) (This is the same as content, but it uses sentiment)
- LDA recommender system
- Multi Armed Bandit and bayesian MAB Recommender
 - Q learning recommender

Support functions
- various methods for document embeddings (doc2vec, lsa, autoencoders)
