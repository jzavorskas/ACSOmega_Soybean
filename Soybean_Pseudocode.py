# -*- coding: utf-8 -*-
"""

Pseudocode for Inverse Design Random Forest/Genetic Algorithm.

"""
# %% Part 1, Initial Definitions and Groundwork
"""
a) Import SoyNAM BLUP Data

Using separate code included in supplemental files, Soybean phenotype and
genotype data was extracted using the R package SoyNAM.

This section imports the genotype, as well as a few different forms of the
phenotype, which will each be used in various sections.
"""
# %%
"""
b) Use GA to optimize UMAP mapping based on Random Forest accuracy.
    
** Fitness Function:
    Train the UMAP reducer with the parameters in the current solution.
    Train the random forest with 80% of UMAP-embedded data (withhold 20%).
    Predict the withheld UMAP-embedded points
    Calculate 1/MAPE for fitness (higher is better)
    
** Define allowed values for:
    UMAP: Number of neighbors
    UMAP: Minimum distance between points
    UMAP: Learning rate
    UMAP: Local Connectivity between points
    UMAP: Repulsion strength of non-correlated points
    UMAP: Negative sample rate (how many points to repulse per epoch)
    RF: Number of estimators
    
** Define verbose functions:
    1) Provide progress updates on best fitness and population each generation.
    2) Display initial population at the beginning of run.
    
** Initialize GA instance and define all parameters:
    Number of generations
    Number of best individuals to perform crossover
    Populations Size
    Number of Genes (how many parameters are being tuned)
    Data type of each parameter
    Allowed range for random initialization of parameters
    How many solutions to roll over to next generation
    Crossover Probabilty and type
    Mutation Probability
    Ask GA to save best solution from each generation
    
** Run GA!

While this will not be included in the manuscript, code is available to
graph each of the available phenotypes against the UMAP-embedded genotype data
using the trained UMAP reducer. The parameters for the trained reducer are
included at Line 271.

"""

# %% 

"""
c) K-Fold Cross Validation of Random Forest
    
    Height and Protein 5-Fold Cross Validation:
        Average Fold Error (MAPE): ~
    
We have decided against using data weighting as it provides marginal improvement.
However, it does improve prediction accuracy at extremes of phenotype data range.
We've left the code in for those who want to try it, but to more reliably
improve the accuracy for points at the extremes, we suggest XGBoost.
"""

# %% Part 2, Problem Statement: Can an individual be regenerated from a neighbor
   # within the population?
   
"""
Part 2:
    
a) Define maximum allowed SNPs.
Find "neighbors" for each genotype within SNP tolerance:
    
for Each Member of Population:
    Check all other members for neighbors within SNP tolerance.
    Save all neighbors for current genotype in a list.
^----

"""


# %%
"""
Part 2b) Experiment to find if RF/GA can directly reproduce a genotype based on
   its phenotype. (Many-to-one problem statement)
   
** Select an individual known to have at least one neighbor in SNP constraints.

** Save the individual's phenotype to use as GA target.

** Delete the individual so the pipeline runs without knowledge of its 
    phenotype or genotype. 

** Define all parameters for the GA:
    Population Size
    Number of Generations
    Max Number of SNPs allowed
    Acceptable Gene Values (0, 1, or 2)
    
** Define the fitness function for the GA:
    
    1) if the individual has more SNPs than allowed:
            remove from population and replace with a known individual
            
    2) if the individual is exactly the same as a member of the population
            assign Fitness = 0, forcing replacement by mutation and crossover
            
    3) otherwise, use the random forest to predict phenotype, and calculate
       fitness based on error from desired value.
    
** Define verbose functions that will display progress updates at the beginning
   of each run and after each generation.
   
** Initialize an instance of the GA as a Python object, adding in a few more
   parameters:
       Number of Elite solutions to keep unchanged
       Number of best individuals considered for crossover
       Data type for all genes in solutions (int)
       Probabilty of crossover
       Type of crossover
       Probability of mutation
       
** Run the GA, saving the current time. Save the following:
    Best all-time individual
    Fitnesses of final generation
    Solutions in final generation

for All Solution in final generation:
    Save solutions with fitness greater than a certain threshold (0.05% error)
    Compare saved solutions with deleted target solution:
        Cosine similarity
        Hamming distance

Observe results. 
*Likely to fail due to many-to-one problem.*

"""

# %% Part 3) Is all Lost? Reframing the Problem.

"""
a) Despite many-to-one constraints, is the pipeline operating as expected?
    (i.e. is it possible to perform this analysis in small steps from known
       genotypes and phenotypes to novel ones?)
    
** Separate all SoyNAM data into three bins:
    10% of the range of phenotype data surrounding target
    All Higher Values
    All Lower Values

** for Each point in each bin:
    calculate cosine similarity to GA individuals
    calculate hamming distance to GA individuals

Observe results. GA individuals are often created from "parents" within the
population whose phenotype is within the same bin. There are exceptions, though!
This provides us the opportunity to hypothetically make large phenotype leaps
with few SNPs.

b) Visualization of Many-to-One Problem:
    
** Retrain UMAP reducer object using newly generated GA points.

** Plot the embedded GA points accentuated from the rest of the population.

Note that all points appear to cluster in various parts of the genotype space.
Figure X.

"""

# %% Part 4) Incremental Optimization Proof of Concept
"""
With one-step inverse design out of the question, a proof of concept of
'incremental' design by "Design, Build, Test, Learn" is necessary. It is
prohibitively difficult to regenerate a genotype using the full search space,
but is it possible if using a large cluster of neighbors?
"""

# %% 

"""
Part 4a) Single Run Incremental Optimization
    
** Initial Definitions:
    Number of Generations (much higher to allow time to explore)
    Population Size
    Genotype Target ID
    Populations (initial and pool to pull from during; cannot use Target)
    
** Save phenotype and genotype of target delete from population.

** Fitness Function: (Same as Part 2b with minor changes)

    1) if the individual has more SNPs than allowed:
            remove from population and replace with a known individual
            
    2) if the individual is exactly the same as a member of the population
            assign Fitness = 0, forcing replacement by mutation and crossover
            
    3) otherwise, use the random forest to predict phenotype, and calculate
       fitness based on error from desired value.

        a) if exact phenotype found, mark with termination fitness (100000)
    
        b) otherwise, calculate fitness normally
    
** Define Verbose on_generation Function: (Same as Part 2b, except with Redundancy)
    
    Every time the best fitness value does not change, increment redundancy.
    
    Stop conditions:
        1) The best solution has been redundant for 300 generations
        2) The termination fitness (100000) has been assigned

    The stop conditions are defined in "on_generation" so the GA does not
    terminate mid-generation.
    
** Initialize GA object.

    All settings are the same as part 2b).
    
** Run GA, saving the current time. Save the following:
    Best all-time individual
    Fitnesses of final generation
    Solutions in final generation
    
** Save any individuals whose phenotype error is <2%. This should grab most
    solutions that are reasonably close.
    
Observe results. Many of the targets within the neighbor cluster I am using:
{1215,2422,2433,2536,3164,3168,3172,3206,3720,3855,3912,4028,4446,4497,
 4729,4969,4993,5447,5478} can be found from the others. The value that comes
with this file, 2422, can be found relatively quickly. The analysis below will
show which neighbors can and cannot be found. The many-to-one problem is likely
to blame for those that cannot be found, suggesting that a local optima is found
on the way to the phenotype of those targets.
"""

# %%
"""
Part 4b) Try all neighbors in 19-neighbor cluster. Can they be found by
           inverse design?
           
** Initial Definitions
    How many iterations allowed to find each target
    Best fitness across all iterations for target
    Lowest number of SNPs between a target and previous closest phenotypes
    SNPs between target and individual with all time closest phenotype
    List to hold lowest and closest phenotype SNPs for each target
    Counter to display which target currently running
    Array of target IDs
    Save current time
    
** Master Loop:
    
for each Target:
    
    Reset tracker values
    
    Find the value the random forest expects, save as target for GA
    Save genotype
    
    Delete the target genotype and phenotype
    
    Define populations (initial and pool to choose from, exclude Target)
    
    for each Iteration:
        
        run GA exactly as in Part 4a)
    
        Save the following:
            Best fitness in final generation
            Genotype of best fitness
            Hamming distance of that genotype from target
            
        Use a few checks:
            
            1) If AllTimeBest exceeded, update best fitness and its hamming.
    
            2) If lowest SNPs exceeded, update. This will always occur on the
                first generation.
                
            3) If termination fitness value passed, update tracker lists and
                terminate loop.
    
    ** If an update to the tracker lists has not yet occurred, update them with
        the lowest SNPs and SNPs to closest phenotype.
        
Observe results. Many of the target values are found exactly, and the others
can be found within a few SNPs. It is possible to perform inverse design by
taking small steps through the genotype search space.
"""

# %% Part 5) Incremental Optimization Case Study

"""
Using more phenotypes and performing in silico "Design, Build, Test, Learn".
"""

# %%

"""
Part 5a) Adding More Phenotypes, Defining Design Goals and Constraints
    
** Retrain on Height (cm), Protein Content (%) and Seed Size (g/100 seeds).
    Error (MAPE): ~1.5%. 
    Height, Protein, Size; henceforth HPS.
    
** Perform 5-fold cross-validation using HPS.
    Average Fold Error (MAPE): ~5%.
    
** Define functions for constraints and fitness functions for each phenotype.
    Goal: Maximize protein content of seeds without encouraging wasteful
          growth (i.e. increasing height or seed size to do so.)
          
    Height:
        1) If above population mean height, introduce penalty.
        2) If below mean, introduce very small reward.
    
    Seed Size:
        1) If above population mean height, introduce penalty.
        2) If below mean, introduce very small reward.
    
    **** The rewards and penalties for Height are lower than seed size because
         height has a much larger range of values than seed size.
         
    Protein:
        1) If below population mean height, introduce penalty.
        2) If above population mean but within 1 standard deviation, small reward.
        3) If between 1 and 2 standard deviations above mean, larger reward.
        4) If more than 2 standard deviations above mean, massive reward.
    
"""

# %%

"""
Part 5b) Incremental Optimization HPS Case Study
    
** Initial Definitions:
    How many cycles of design, build, test, learn (DBTL)?
    How many individuals will be generated in each DBTL cycle.
    
    Population Size
    Redundancy before terminating
    Number of generations
    How many SNPs are allowed?
    
    Reimport genotype and phenotype, in case of any deletions or additions
    
** Fitness Function:
    
    Identical to Part 4b) except:
    
    Split random forest prediction into individual phenotypes and call to
    the custom fitness functions described above.
    
** Verbose Functions:
    
    Identical to Part 4b) except:
    
    The phenotype values for best Height, Protein, Size are reported every 
    generation.
    
    on_start is used here to signal the start of a run GA run.
    
**** Master Loop:
 
for each DBTL Cycle:
    
    Create tracker arrays for genotype and phenotype.
    
    Retrain the random forest with the new genotypes and phenotypes from the
    previous DBTL cycle.
    
    for each GA Run:
        
        Run GA. This GA is identical to Part 4b) except for a few inputs
        as above in "Initial Definitions".
        
        After each run, save:
            Best Individual
            Best Individual's Fitness Value
            Best Individual's Phenotype
            
        Add Best Individual's genotype and phenotype to tracker arrays
        
    
    Add the accumulated genotypes and phenotypes into the population at the
    end of each DBTL cycle. If this was not in silico, these individuals would
    need to be created in real life and confirmed before being added.
            
    
"""

# %% 
"""
Part 5c) Visualization of Results
    
** Create a histogram with 50 linearly space bins between the min and max value
    for protein content:
        Original Data = Green
        GA-Generated = Red
        Bins will stack on top of each other, with green being much thinner
        
Red bars should show a shift of the mean to higher values.

** Find the nearest neighbor of each GA-Generated Individual:
    Create tracker arrays for:
        Index (in original dataset) of neighbor.
        Genotype of neighbor
        SNPs required from neighbor to GA individual
        Difference in phenotype from neighbor to individual
        Difference in phenotype (both measured by RF)
        
This section shows two things: 
    1) significant phenotype steps are being taken
    2) the GA tends to use all or most available SNPs (optimization "corner")
    
** Graph Height vs. Protein and Seed Size vs. Protein:
    

This section is meant to show that the algorithm pushes the boundaries
of phenotype and back up my conceptual map.


    
"""