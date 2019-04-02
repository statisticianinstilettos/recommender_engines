# recommender_engines
WIP

This will be a personal libary for freelance data science work. These classes can be used to train and deploy product recommendations using client provided data. 

Will contain recommender classes the following models:

- Similar items - Content Filtering(matrix factorization) and Deep Content Filtering (autoencoders). Takes in tabular features, and/or document embeddings. 
- People also purchase - Collaborative Filtering (solved via matrix decomposition by gradient descent) and Deep Collaborative Filtering.
- You may also like - Factorization Machines
- Pairs well with - association rules, Deep outfits. 
- Describe what you are looking for bot - Information retrieval (cosine sim with document embeddings. take in any type of document embeddings) (This is the same as content)

Support functions
- various methods for document embeddings (doc2vec, lsa)
