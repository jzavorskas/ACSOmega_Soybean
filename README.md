# ACSOmega_Soybean
Companion code for https://doi.org/10.1021/acsomega.4c01704.

This code is organized into five parts, which represent different phases of the project:

## Part 1:
a) Data is imported and preprocessed to prepare for use for both random forest regression and dimensionality reduction/clustering by Uniform Manifold Approximation and Projection (UMAP).
b) Data is clustered in a reduced-dimensional space, with coloration to help visualize the clustering of various phenotypes. 
c) K-fold cross validation is performed (with and without sample weighting) to  validate random forest. Random grid hyperparameter tuning is also included, which uses sequential k-fold cross validation to select the best set of hyperparameters.

## Part 2:
a) All genotypes with neighbors within 20 SNPs (Hamming distance) are found.
b) To define the fundamental problem with inverse design (one-to-many), we attempt to recreate one of the genotypes known to have many neighbors. We use a genetic algorithm whose fitness function (objective) is to minimize the difference between the known phenotype and the phenotype of the current solution. Phenotypes of each solution are predicted via a random forest trained on all genotype/phenotype pairs. The GA is not able to do so; it gets stuck in local minima, regardless of the fact the the genotype/phenotype pair selected has many nearby neighbors known to the RF.

## Part 3:
a) This section is all about understanding the problem defined above. First, we check if the final GA population's genotypes are close to the known neighbors from part 2. Are the genotypes found drastically different, or close?
b) Next, we use UMAP to visualize the genotypes of each individual from the final GA population against the genotypes of the full population. The UMAP genotypes of the GA population are spread out across the entire latent space, indicating their genotypes are very different.
c) Finally, we use UMAP to visualize and compare the genotypes of the final GA population to just the individuals known to be neighbors of the target genotype. The GA population does not necessarily cluster around those genotypes.

## Part 4:
a) As a proof of concept, we attempt to recreate the target genotype given a starting population of only its neighbors. This is significantly more successful, but one-to-many still causes trouble.
b) Repeating the proof of concept for all neighbors in the cluster.

## Part 5:
a) Choosing a phenotype to be optimized incrementally, we choose protein content and preprocess the data so only genotypes and protein content phenotype remain.
b) Main incremental optimization loop, in which a genetic algorithm generates an individual with locally optimal phenotype. This individual is then added to the population, and the random forest is retrained to add the individual. The phenotype is incrementally optimized as new locally optimal phenotypes are generated.
c) Visualization of phenotypes and genotypes from this incremental optimization process.
