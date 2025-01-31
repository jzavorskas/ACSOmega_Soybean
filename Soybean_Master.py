# -*- coding: utf-8 -*-
"""
Start Date: 2/02/2023
Last Edit: 5/16/2023

In this script, I will be using visualizing the SoyNAM soybean dataset 
with the uniform manifold approximation and project (UMAP) algorithm.

Then, without UMAP dimensionality reduction, a random forest (RF) will be trained to 
predict a set of phenotypes from the genotype data available. A genetic 
algorithm (GA) will then be used to try to produce novel phenotypes by using the
trained RF as a forward simulator. The GA's solutions will be genotype with the
same format as the SoyNAM set, and will progress until a genotype that provides
the desired phenotype is discovered. The solution space of this problem is
massive, and will likely require an explorative, incremental approach similar
to the "Design, Build, Test, Learn" (DBTL) paradigm.

"""
# %% Table of Contents

"""
##########################################
Part 1: Definitions and Visualization
a) Import and Preprocessing
b) UMAP Optimization by GA; Visualization of Phenotypes
c) Random Forest Validation; K-Fold Cross Validation
_______________________________________________
Part 2: Problem Statement
a) Maximum SNP Tolerance Definition; Find Genotypes with Neighbors
b) Can GA/RF find a removed point which is known to have nearest neighbors?
_______________________________________________
Part 3: Visualization and "Sanity Check" of Pipeline
a) Metrics by Bin: Generated Individual close to Genotypes of similar phenotype?
b) UMAP Visualization of Generated Individuals against full population's phenotypes.
c) UMAP Visualization of Generated Individuals against target neighbor cluster.
_______________________________________________
Part 4: Incremental Optimization Proof of Concept
a) GA Restricted to just individuals in a cluster. Can we accurately make an incremental change?
b) In the largest cluster (19 individuals), how many of them can be recreated (reasonable time limit)?
_______________________________________________
Part 5: Incremental Optimization Case Study
a) Choose phenotype(s) to optimize.
b) Loop incremental optimization, adding new individuals to the pipeline along the way.
c) Visualize progression of individuals toward the phenotype goal.

"""

# %% Step 1a) Import necessary packages.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import umap
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import pygad
import time
import scipy
import scipy.signal
import random
import csv
import ast
import statistics
import math

# %% Part 1a) Define Functions for use below:
# Scikit learn only has MAPE in 0.24, which is not available in conda.
# Easier to write my own than to try to update.

# This function calculates the mean absolute percentage deviation between a
# predicted set of data and the "true" data.
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = np.abs(y_true-y_pred)
    mape = 100 * np.mean(errors/y_true)
    return mape

def GenImport():
    GenFrame = pd.read_csv("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\HeightGen.csv",sep=",")
    # Change SNP Array over to an (numpy) array.
    Gen = GenFrame.to_numpy()
    # Remove the extra identifiers from the dataframe.
    Gen = np.delete(Gen,0,1)
    # Remove the final SNP of the dataset to bring to an even number.
    Gen = np.delete(Gen,4400,1)
    return Gen

def PhenImport():
    # Bring in phenotype data for all relevant characteristics.
    HeightPhen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\HeightPhen.csv")
    OilPhen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\OilPhen.csv")
    ProteinPhen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\ProteinPhen.csv")
    SizePhen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\SizePhen.csv")
    
    # Stack phenotypes together for use in other functions.
    Phen = np.column_stack((HeightPhen,OilPhen,ProteinPhen,SizePhen))
    HeightProtein = np.column_stack((HeightPhen,ProteinPhen))
    HeightProteinSize = np.column_stack((HeightPhen,ProteinPhen,SizePhen))
    
    return HeightPhen, HeightProtein, HeightProteinSize, Phen

# %% Part 1a) Import One-hot encoded data from files.
# These files have been extracted from the SoyNAM Package in R.

global Gen
global HeightProtein

# Bring in phenotype data for all relevant characteristics.
HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
# Import the genotype (SNP Array).
Gen = GenImport()

# Delete the phenotype outlier from analysis.
HeightPhen = np.delete(HeightPhen,2459,axis=0)
HeightProtein = np.delete(HeightProtein,2459,axis=0)
HeightProteinSize = np.delete(HeightProteinSize,2459,axis=0)
Phen = np.delete(Phen,2459,axis=0)
Gen = np.delete(Gen,2459,axis=0)

# %% Part 1b) GA #1: Fitness Function for GA that tunes UMAP/RF together.

def fitness_function1(sol,sol_idx):
    # Create a UMAP transformer (reducer) based on the parameters input
    # by the genetic algorithm.
    reducer = umap.UMAP(n_neighbors=sol[0],metric="cosine",
                        n_epochs=100,min_dist=sol[1],learning_rate=sol[2],
                        local_connectivity=sol[3],repulsion_strength=sol[4],
                        negative_sample_rate=sol[5])
    
    # Embed all the data into the lower dimensional space.
    embedding = reducer.fit_transform(Gen)
    
    # Split the testing data up, I'm using 9 as the random seed because it's my favorite number :)
    x_train, x_test, y_train, y_test = train_test_split(embedding,HeightPhen,test_size=0.2,random_state=9)
    
    # Create a random forest regressor based on the final GA parameter, and
    # fit it such that it predicts phenotype from reduced genetic data.
    regressor = RandomForestRegressor(n_estimators=sol[6])
    regressor.fit(x_train,y_train)
    
    # Use the random forest to predict the withheld testing data.
    EmbedPredict = regressor.predict(x_test)
    
    # Using Mean Absolute Percentage Error as the testing function.
    # However, the GA package is coded to find the HIGHEST number for fitness, 
    # so I must take the inverse of MAPE, as lower MAPE is actually better.
    fitness = 1/MAPE(y_test,EmbedPredict)
    return fitness

# %% Part 1b) Notes Section
# Genes:
    # 1 - (UMAP) Number of Nearest Neighbors. [n_neighbors]
    # 2 - (UMAP) Minimum Distance between embedding points. [min_dist]
    # 3 - (UMAP) Learning Rate during Optimization. [learning_rate]
    # 4 - (UMAP) How many points are connected when positions are moved. 
    #            [local_connectivity]
    # 5 - (UMAP) Strength of negative interaction between dissimilar points.
    #            [repulsion_strength]
    # 6 - (UMAP) Number Dissimilar points considered for repulsion per point.
    #            [negative_sample_rate]
    # 7 - (RF) Number of Decision Trees per ensemble. [n_estimators]
    # 8 - (RF) Maximum Depth of Each Tree. [max_depth]
    # 9 - (RF) Maximum Number of Features considered when making splits.
    #          [max_features]
    # 10 - (RF) Minimum number of samples required to create a leaf.
    #           [min_samples_leaf]
    # 11 - (RF) Minimum number of samples required to create a new split.
    #           [min_samples_split]

# %% Part 1b) GA #1: Define Allowed values, verbose functions, GA instance.

# This parameter defines a range of discrete values allowed for parameters
# and the step in between the values. 

gene_space1 = [
              {'low': 10, 'high': 50}, # n_neighbors
              {'low': 0.025, 'high': 0.5}, # min_dist
              {'low': 0.25, 'high': 5}, # learning_rate
              {'low': 1, 'high': 15}, # local_connectivity
              {'low': 0.25, 'high': 10}, # repulsion_strength
              {'low': 3, 'high': 30}, # negative_sample_rate
              {'low': 25, 'high': 1000}, # n_estimators (RF)
              ]


### Step 1 GA: Define verbose functions.
# I want to report each time fitness func finishes and a generation completes.
    
# The GA will automatically print all of the following information after each 
# generation.
def on_generation1(ga_instance):
    global StartTime
    print("Generation: ", ga_instance.generations_completed)
    print("Fitness of the Best Solution :", ga_instance.best_solution()[1])
    print("Time Elapsed: ", (time.time()-StartTime))
    print("Population:")
    print(ga_instance.population)
  
# Before calculating fitnesses of the first generation, the GA will display
# the parameters each individual in the initial population will use.    
def on_start1(ga_instance):
    print("Initial Population:")
    print(ga_instance.population)
    print("Time Elapsed: ", (time.time()-StartTime))


# %% Part 1b) Genetic Algorithm to Simultaneously Tune UMAP/RF parameters.
# I'm not sure yet whether this is a good idea, but tuning seems to be
# the way forward. I'll tune some common/influential UMAP and RF params
# and see how that effects MSE and MAPE.

ga_instance1 = pygad.GA(# How many GA iterations?
                        num_generations=8,
                        # Each generation how many individuals will cross over?
                        num_parents_mating=4,
                        # See above for my fitness function.
                        fitness_func=fitness_function1,
                        # Population Size of the GA: how many parameter sets?
                        sol_per_pop=15,
                        # I'm optimizing 7 parameters.
                        num_genes=7,
                        # Data types of each parameter. 
                        # Position here = position in GA.
                        gene_type = [int,float,float,int,float,int,int],
                        # GA randomly picks initial values. 
                        # These are the allowed range.
                        init_range_low = 0,
                        init_range_high = 1000,
                        # How many individuals kept unchanged between generations.
                        keep_elitism = 1,
                        crossover_probability = 0.7,
                        # Every point in the array has a chance to cross over.
                        crossover_type = "scattered",
                        # This probability is relatively high, but the search space is large.
                        mutation_probability = 0.2,
                       
                        # See above for definitions of these:
                        gene_space=gene_space1,
                        on_generation=on_generation1,
                        on_start=on_start1,
                       
                        # GA automatically saves an array of the best solution 
                        # every generation. Makes it easier to visualize the best
                        # solution later.
                        save_best_solutions=True
                        )
   
# %% Part 1b) GA #1: Run the GA instance!     

global StartTime
StartTime = time.time()

ga_instance1.run()

# %% Part 1b) UMAP: create reducer object, used for visualization.

# All parameters shown here are described in the GA notes section above.
# These parameters were selected based on the reduced space that produced the best
# random forest prediction accuracy of phenotype.
reducer = umap.UMAP(n_neighbors=92,metric="cosine",
                    n_epochs=100,min_dist=0.075,learning_rate=1.36,
                    local_connectivity=5,repulsion_strength=2.76,
                    negative_sample_rate=24)

# reducer = umap.UMAP()

# Embed the genotype data into a lower dimensional space.
# Genotypes that cluster togeter have similar genotypes.
embedding = reducer.fit_transform(Gen)

# %% Part 1b) Plots: Visualization of Genotype in 2-D.

# Four continuous phenotypes extracted from the SoyNAM dataset.
PhenType = ["Height", "Oil", "Protein", "Size"]
PhenDescription = ["Height in cm", "Oil Content (%/mass)", "Protein Content (%/mass)",
                    "Seed Size (g/100 seeds)"]
# Create four subplots. The axis object will be used to assign data to each.
fig, ax = plt.subplots(2,2,figsize=(13,13))
# Iterator to keep track of which phenotype (and axis) is being plotted.
Count = 0

# The subplots are in a 2x2 grid. These loops iterate over each subplot.
for i in range(2):
    for j in range(2):
        
        # Each of the four subplots plots the exact same genotype data,
        # which comes from UMAP's embedding. Instead of 4400 SNPs, 2 components.
        scatter = ax[i][j].scatter(embedding[:,0],
                                  embedding[:,1],s=2,c=Phen[:,Count],
                                  cmap='viridis')
        
        # The genotype is then colored by one of the four phenotypes.
        ax[i][j].set_title(("Color:" + PhenType[Count]))
        ax[i][j].set_xlabel("UMAP Coordinate 1")
        ax[i][j].set_ylabel("UMAP Coordinate 2")
        
        # Create a colorbar next to the current subplot which acts as a heatmap
        # to describe the color of each "genotype point."
        cbar = fig.colorbar(scatter,ax=ax[i][j],label=PhenDescription[Count])

        # Increment so the next loop iteration uses the next phenotype.
        Count += 1

plt.show()

# This code will plot a full-size graph of height data ONLY.
# (Dimensionally reduced genotype clustered by height value):

# plt.scatter(embedding[:,0],embedding[:,1],
#             s=2,c=HeightPhen)
# plt.colorbar()

# %% Part 1c) RF: Regression of UMAP data.

# This standalone function is for validation after GA runs ONLY, so see below
# for a fully commented version of this code.

# Splitting a single phenotype run:
# x_train, x_test, y_train, y_test = train_test_split(Gen,HeightPhen,test_size=0.2)
# Splitting both Height and Protein phenotypes:
x_train, x_test, y_train, y_test = train_test_split(Gen,HeightProtein,test_size=0.2)
# Splitting all available phenotypes:
# x_train, x_test, y_train, y_test = train_test_split(Gen,HeightProtein,test_size=0.2)

# Initialize the RF object. For this project we use default parameters because
# tuning gave marginal performance boosts at best.
regressor = RandomForestRegressor()

# If doing K-Fold Validation or excluding certain data, use below:
# regressor.fit(x_train,y_train)

# If using a GA, use all of the data to train RF:
    # Sample weight can be included by running the below sections and removing
    # the commented section below.
regressor.fit(Gen,HeightProtein,#sample_weight=sample_weight
              )

# Predict the held-out genotype/phenotype data. If x_train/y_train used to
# train regressor, this is true accuracy. If x_test is represented in training
# set, this is in-bag accuracy.
Predict = regressor.predict(x_test)

# Functions for assessing the accuracy of the RF.
# Mean Squared Error and Mean Average Percentage Error.
MSE = mean_squared_error(y_test,Predict)
MeanAPE = MAPE(y_test,Predict)

# %% Part 1c) RF: K-Fold Cross Validation

# How many sections should the data be split into?
Folds = 5

# This object will split any dataset fed to its .split() method into n_splits
# parts by index.
kf = KFold(n_splits=Folds, random_state=None)
# Accuracy scores will be stored in this array.
MAPEFold = []
# Initializing this variable to make KFold verbose.
FoldCount = 0

for train_index, test_index in kf.split(Gen):
    # Partition the data using the indexes provided by KFold.split(). 
    # IMPORTANT: to maintain accuracy, must also split and use sample weights!
    x_train, x_test = Gen[train_index,:], Gen[test_index,:]
    y_train, y_test = HeightProtein[train_index], HeightProtein[test_index]
    # sample_test = sample_weight[train_index]

    # Create a fresh instance of the regressor each iteration, and train it.
    FoldRegressor = RandomForestRegressor()    
    FoldRegressor.fit(x_train,y_train,# sample_weight=sample_test
                      )

    # Predict the phenotypes of all the withheld genotype information.
    # Then, calculate the MAPE as an accuracy measurement.
    FoldPredict = FoldRegressor.predict(x_test)
    MAPEFold.append(MAPE(y_test,FoldPredict))
    
    # Verbose section. How many folds have been completed?
    FoldCount += 1
    print("Fold Completed: {}".format(FoldCount))

# Calculate and print the average MAPE during KFold.
AvgMAPE = sum(MAPEFold)/Folds
print("Average MAPE: ", AvgMAPE)

# %% Part 1c) RF: Data Weighting, count how many samples in each bin.

# Initialize a dictionary with keys representing height bins of approximately
# 10 cm each. 
HeightDict = {"63-75":0,"75-85":0,"85-95":0,"95-105":0,
              "105-115":0,"115-125":0,"125-136":0}

# The following loop counts the number of genotypes whose phenotype is within
# the initialized bins. There is a more efficient way to do this!
for data in HeightPhen:
    if data < 75:
        HeightDict["63-75"] += 1
    elif data >= 75 and data < 85:
        HeightDict["75-85"] += 1
    elif data >= 85 and data < 95:
        HeightDict["85-95"] += 1
    elif data >= 95 and data < 105:
        HeightDict["95-105"] += 1  
    elif data >= 105 and data < 115:
        HeightDict["105-115"] += 1
    elif data >= 115 and data < 125:
        HeightDict["115-125"] += 1
    else:
        HeightDict["125-136"] += 1
      
# Initialize a list with spaces for the number of bins.
Proportions = []

# Using the counted phenotypes, calculate the proportion of total phenotypes
# each bin represents.
for key in list(HeightDict):
    Proportions.append(5487/HeightDict[key])


# Normalize the calculated proportions such that the least abundant group of
# phenotypes has a value of 1. More abundant phenotypes should have lower
# values now.
Proportions = [x/max(Proportions) for x in Proportions]

# %% Part 1c) RF: Use Proportions to Define Sample weights.

sample_weight = np.zeros((5487))
Iterable = 0

# Based on the above calculations, assign individual weights to each sample
# in the dataset. Samples closer to the population mean should have 
# significantly lower weights. The goal is to remove bias toward more
# abundant samples and explore the extremes of the phenotypes.
for data in HeightPhen:
    if data < 75:
        sample_weight[Iterable] = Proportions[0]
    elif data >= 75 and data < 85:
        sample_weight[Iterable] = Proportions[1]
    elif data >= 85 and data < 95:
        sample_weight[Iterable] = Proportions[2]
    elif data >= 95 and data < 105:
        sample_weight[Iterable] = Proportions[3] 
    elif data >= 105 and data < 115:
        sample_weight[Iterable] = Proportions[4]
    elif data >= 115 and data < 125:
        sample_weight[Iterable] = Proportions[5]
    else:
        sample_weight[Iterable] = Proportions[6]
    Iterable += 1

# %% Part 1c) K-Fold Cross Validation Tuning (Random Grid)

# While we do not include a tuned RF in our pipeline, we did consider the
# effects of tuning, shown below. Minimal improvements on RF accuracy occur,
# while increasing training time, so tuning has been omitted.

# How many trees will be included in each population?
n_estimators = [int(x) for x in np.linspace(start=100,stop=1500,num=15)]
# How many features will be included for consideration? 0: all features,
# sqrt: sqrt(total number of features)
max_features = [0, 'sqrt']
# Max depth ~= number of splits ~= number of features considered per tree.
# Can also allow trees to have an unlimited depth.
max_depth = [int(x) for x in np.linspace(start=10,stop=110,num=11)]
max_depth.append(None)
# When trees are created, what is the fewest number of training samples
# they are allowed to assign to one side of the split?
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]

# Assemble all tuning parameters and their allowed values into a dictionary.
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Initialize the RandomizedSearchCV object with the defined options.
# ** Verbose = 4 forces all available progress messages.
# ** n_jobs = -1 uses all available processors.
rf_random = RandomizedSearchCV(estimator=regressor, param_distributions=random_grid,
                                scoring=make_scorer(MAPE, greater_is_better=False),
                                n_iter=25,cv=Folds,verbose=4,n_jobs=-1)

rf_random.fit(Gen,HeightPhen)

# %% Part 2) Problem Statement
"""
In inverse design, the greatest challenge is the inherent "many-to-one" nature
of inverse function mapping. This challenge is exacerbated by datasets with
a large number of features. 

Due to concerns about viability, feasibility of creation, and having a search
space that is too large, we have decided to limit the inverse design pipeline
to individuals that require less than 20 SNPs from an existing individual.
This could even be too lenient a constraint, given biological restrictions.

In Section 2, we aim to show first that it is prohibitively difficult to 
reconstruct a point that has been removed from the population using our
20 SNP constraint, because some individuals simply do not have neighbors
with 20 SNPs. We therefore will begin by selecting a point that is known to 
have neighbors with 20 SNPs and show that it is still prohibitively difficult 
to reach one of those points due to the "many-to-one" nature of mapping 
phenotype to genotype.

"""

# %% Part 2a) Neighbors: Find genotypes that differ by <= 20 SNPs.
    
# How many SNPs are individuals generated by the GA allowed to have?
SNPTol = 20

# Initialize an array that will hold a new population:
# all individuals that have a neighbor within the SNP tolerance.
NeighborArray = np.zeros((1,4400))
# This dictionary is an index reference between the full genotype array (Gen)
# and the new NeighborArray. Keys will be the new index, and will point to
# the old index.
NeighborDict = {}
# This dictionary keeps track of each individual checked in Gen and which other
# are within the SNP tolerance.
# Keys: Genotype of interest. Pointing to: List of indexes of close genotypes.
WhatNeighbor = {}

# Keep track of how many individuals are picked as having a close neighbor.
# This counter will be the keys in NeighborDict.
Picked = 0

# idx will keep track of the individual we are checking all others against.
    # it will therefore be the "Old index" in NeighborDict and the keys in
    # WhatNeighbor
    
# idy will keep track of the inner loop. These indexes will be part of the
# lists in WhatNeighbor.

for idx, Genotype in enumerate(Gen):
    
    NeighborList = []
    HasNeighbor = False
    
    for idy, Checktype in enumerate(Gen):
        
        # Avoid checking the genotype against itself.
        if idx != idy:
            # Calculate the Hamming distance to the currect check points.
            Distance = scipy.spatial.distance.hamming(Genotype,Checktype)
        else:
            Distance = 0
        
        # Distance != 0 may not be0 necessary because of idx != idy.
        # This is just in case there are duplicates in the dataset.
        if Distance != 0 and Distance <= SNPTol/len(Gen[0,:]) and not HasNeighbor:
            # Add the current "big loop" genotype to NeighborArray,
            # indicating that it has at least one neighbor.
            NeighborArray = np.append(NeighborArray,Genotype)
            # Use the picked iterator in NeighborDict. This will keep track of
            # how many genotypes have been identified and keep track of their
            # original index in Gen.
            NeighborDict[Picked] = idx
            # Finally, add the index of the "small loop" checked genotype to
            # a list. It is possible that a genotype has multiple acceptable
            # neighbors. This will be handled below.
            NeighborList.append(idy)
            # This complicated loop step only needs to happen once. Once it 
            # does, only new neighbors need to be tracked by NeighborList.
            # This boolean indicates whether a neighbor has already been found.
            HasNeighbor = True
            Picked += 1
        elif Distance != 0 and Distance <= SNPTol/len(Gen[0,:]) and HasNeighbor:
            # A neighbor has already been found. Therefore, the "big loop"
            # genotype has already been document. Only need to document the
            # new neighbor here.
            NeighborList.append(idy)
    
    # Check if NeighborList is empty. bool([]) = False.
    if bool(NeighborList):
        # Assign the list of neighbors to the current "big loop" genotype.
        WhatNeighbor[idx] = NeighborList
        
    # Verbose section. Reports every 25th genotype completed.
    if idx % 25 == 0:
        print("Genotype Completed: ", idx)

# NeighborArray = np.delete(NeighborArray,0,0)

# %% Part 2a) Neighbors: Save Neighbor Data
# Save WhatNeightbor, which contains all data needed to regenerate the
# above analysis.

with open('C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\WhatNeighbor.csv', 
          'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=WhatNeighbor.keys())
    writer.writeheader()
    writer.writerow(WhatNeighbor)

# %% Part 2a) Import Saved Neighbor Data
# Come back to this. Need to deal with pandas reading all as string.
LoadFrame = pd.read_csv('C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\WhatNeighbor.csv')
WhatNeighborIn = LoadFrame.to_dict()
WhatNeighbor = {}
NeighborArray = np.zeros((1,4400))
NeighborDict = {}

DictCount = 0

for key in WhatNeighborIn.keys():
    NeighborHold = WhatNeighborIn[key]
    WhatNeighbor[int(key)] = ast.literal_eval(NeighborHold[0])
    NeighborArray = np.append(NeighborArray,Gen[int(key),:].reshape(1,-1),0)
    NeighborDict[DictCount] = int(key)
    
    DictCount += 1

NeighborArray = np.delete(NeighborArray,0,0)

# %% Part 2b) GA #2: Proof of Problem; Initial Definition/Remove a Point

global DesiredValue
global NumGen2
global PopSize2

# In order to ensure only a single point is deleted, reimport Gen everytime.
# Go to line 110!!!!!!

# Select an individual known to have a large cluster of neighbors within the
# population. 
OriginalID = 2422

# Target value for the genetic algorithm (GA).
# DesiredValue = HeightProtein[OriginalID,:]

# What does the random forest think our desired point's value is?
    # This is the only way we will be able to regenerate the exact genotype.
    # In the true pipeline, the data will be seen through the lens of the RF, 
    # it would be an unrealistic proof of concept to try to generate the real value.
DesiredValue = regressor.predict(Gen[OriginalID,:].reshape(1,-1))

# Hold the deleted individual from Gen for analysis.
DeletedGen = Gen[OriginalID,:]

# Delete the value we're trying to recreate from the population.
Gen = np.delete(Gen,OriginalID,0)
HeightProtein = np.delete(HeightProtein,OriginalID,0)

# %% Part 2b) GA #2: Parameter Definition

global Redundancy2

# Size of the group of heights which will be considered "close enough" to our
# desired height. This will be used to validate the GA is creating reasonable
# individuals.
BinSize = 5
# Percentage of data range that will make up the "close enough" bin. 
# e.g. Height range is 65 to 135 cm, so the "close enough" bin is 7 cm wide.
BinTolerance = 0.1

# How many genotypes will be considered each generation by the GA?
PopSize2 = 20
# Number of generations the GA will run for.
NumGen2 = 50
# How many changes will me allow away from a known individual?
SNPTol = 20
# How many "genes" should the GA allow in each solution?
NumGenes = 4400
# How many repeated best solutions should be allowed?
Redundancy2 = 10

# This parameter defines the square region that represents the "interpolation"
# region of the UMAP embedding.

# Gene_space limits the inputs that can be used for genotype. Because the 
# data consists of one-hot encoded genotype by:
    # 0 - no mutation from reference
    # 1 - heterozygous (single allele) mutation from reference
    # 2 - homozygous (both alleles) mutation from reference
gene_space2 = [0, 1, 2]

# %% Part 2b) GA #2: Fitness Function for Full Population Inverse Design.

# At this step, the UMAP embedding already exists, and the RF is already
# trained. The GA will initialize individuals in the low-dimensional space and
# find their heights using GA. The individuals with height closest to the
# desired value will be selected.

def fitness_function2(sol,sol_idx):

    # Preallocate a vector with the same length as the number of genotypes
    # available in the data.
    DistanceHold = np.zeros((len(Gen[:,0]),1))
    # Calculate the Hamming distance (proportion of mismatches) between 
    # the current GA solution and every other point in the data.
    for idx, TestPoint in enumerate(Gen):
        DistanceHold[idx,:] = scipy.spatial.distance.hamming(TestPoint,sol)
    
    # If there are more SNPs in the current solution than the set tolerance
    # allows, throw out the solution by replacing it with a genotype from the
    # dataset.
    if min(DistanceHold) > SNPTol/len(Gen[0,:]):
        if ga_instance2.generations_completed >= NumGen2 - 1:
            pass
        else:    
            RandPop = random.randrange(len(Gen))
            ga_instance2.population[sol_idx,:] = Gen[RandPop,:]
    # If the current solution is a genotype that exists in the dataset, set
    # its fitness to zero. This ensures that it will not be in the next
    # generation's population. However, because dataset genotypes introduced by
    # the "if" statement will exist in the population for one generation, their
    # genotypes will still have the chance to mate and mutate! Win/win.
    elif min(DistanceHold) == 0:
        return 0
        
    # Use the random forest to predict the phenotype of the each member of
    # the population.
    Predict = regressor.predict(sol.reshape(1,-1))    
    # Would it be better to use MAPE or MSE here?
    # The GA package is coded to find the HIGHEST number for fitness, 
    # so I must take the inverse of MAPE, as lower MAPE is actually better.
    # fitness = 1/MAPE(DesiredValue,Predict)
    if np.sum(abs(DesiredValue-Predict)) == 0:
        fitness = 100000
    else:
        fitness = 1/(np.sum(abs(DesiredValue-Predict)))
    return fitness

# %% Part 2b) GA #2: Definitions of Verbose Functions

# The GA will automatically print all of the following information after each 
# generation.
def on_generation2(ga_instance):
    global StartTime
    global Redundancy
    
    if max(ga_instance.last_generation_fitness) == max(ga_instance.previous_generation_fitness):
        Redundancy += 1
    else:
        Redundancy = 0
        
    if ga_instance4.generations_completed == 0:
        pass
    else:
        if Redundancy >= Redundancy2:
            return "stop"
    
    print("Fitness of Best Solution: ", ga_instance.best_solution()[1])
    print("Time Elapsed: ", (time.time()-StartTime))
    print("Generation: ", ga_instance.generations_completed)
    print("Redundancy: ", Redundancy)
    print("________________________________________________")

        
# Before calculating fitnesses of the first generation, the GA will display
# the parameters each individual in the initial population will use.    
def on_start3(ga_instance):
    # print("Initial Population:")
    # print(ga_instance.population)
    print("Generation Time Elapsed: ", (time.time()-StartTime))
    print("Total Time Elapsed: ", (time.time()-TotalTime))

# %% Part 2b) GA #2: Incremental Optimization Problem Statement

global TotalTime
global Redundancy

ProblemStatementIter = 10
Redundancy = 0
TotalTime = time.time()
BestIndivHold = np.zeros((ProblemStatementIter,4400))    
GenGa = Gen
PopSize = 25

for Iter in range(ProblemStatementIter):
    
    print("Iteration Begin: ", Iter+1)
    
    ga_instance2 = pygad.GA(# How many GA iterations?
                           num_generations=NumGen2,
                           # Each generation how many individuals will cross over?
                           num_parents_mating=5,
                           # See above for my fitness function.
                           fitness_func=fitness_function2,
                           # Population Size of the GA: how many parameter sets?
                           sol_per_pop=PopSize,
                           # I'm optimizing 4400 SNPs, to create an individual
                           # with desired phenotype.
                           num_genes=NumGenes,
                           # Anchor the GA at existing points to ensure valid
                           # results and speed up the process.
                           initial_population = np.random.permutation(Gen)[:PopSize,:],
                           # Data types of each parameter. All should be Int 
                           # (one-hot encoding)
                           gene_type = int,
                           # GA randomly picks initial values. 
                           # These are the allowed range.
                           init_range_low = 0,
                           init_range_high = 2,
                           # How many individuals kept unchanged between generations.
                           keep_elitism = 2,
                           crossover_probability = 0.3,
                           # In two solutions, two points are selected, 
                           # between which all points are switched.
                           crossover_type = "two_points",
                           # I've lowered this probability, because the search space
                           # is relatively small.
                           mutation_probability = 0.002,
                           
                           # See above for definitions of these:
                           gene_space=gene_space2,
                           on_generation=on_generation2,
                           on_start=on_start3,
                           
                           
                           # GA automatically saves an array of the best solution 
                           # every generation. Makes it easier to visualize the best
                           # solution later.
                           save_best_solutions=True,
                           save_solutions=False
                           )

    # global StartTime
    StartTime = time.time()
    
    ga_instance2.run()

    # Find individuals with phenotype close to desired.     
    
    # Pull information about the final population of solutions for visualization
    # and analysis.
    AllFitness = ga_instance2.last_generation_fitness
    BestIndividual = ga_instance2.best_solutions[0]
    LastSolutions = np.array(ga_instance2.solutions[-PopSize-1:-1])   
    
    # Compile all data points together in order; this will be analyzed later!
    GenGa = np.append(GenGa,BestIndividual.reshape(1,-1),axis=0)


# %% Part 2b) Compare individuals found above to known genotype of target.

Metrics = np.zeros((ProblemStatementIter,2))
BigArray = GenGa[-ProblemStatementIter:-1,:]

for GA in BigArray:
    
    Metrics[idy,0] = scipy.spatial.distance.cosine(DeletedGen,GA)
    Metrics[idy,1] = scipy.spatial.distance.hamming(DeletedGen,GA)*4400
    
# %% Part 3a) Partition the data into three bins.
 
# The "In" bin will be smaller,
# representing individuals similar to the desired phenotype. 
# The other bins will be much larger, extending to the extremes of the dataset.

# This will partition the dataset into two large bins and one small bin that
# contains a "BinTolerance" percentage of the data.
LowTol = DesiredValue - (max(HeightPhen)-min(HeightPhen))*BinTolerance
HighTol = DesiredValue + (max(HeightPhen)-min(HeightPhen))*BinTolerance

# Create four cutoff points. Two are the extremes, and two are defined by
# the bin tolerance and the desired value.
Partitions = [min(HeightPhen), LowTol, HighTol, max(HeightPhen)]

# Initialize arrays/lists that will hold the genotype and phenotype data
# after it has been split into three bins.
LowArray = np.zeros((1,2))
InArray = np.zeros((1,2))
HiArray = np.zeros((1,2))
LowGen = np.zeros((1,4400))
InGen = np.zeros((1,4400))
HiGen = np.zeros((1,4400))
LowPhen = []
InPhen = []
HiPhen = []
LowIndex = []
InIndex = []
HiIndex = []
Iterator = 0

# Using the shared indexes of genotype, reduced genotype, and phenotype
# data, split all data into three bins.
for idx, Data in enumerate(HeightPhen):
    if Data <= Partitions[1]:
        LowArray = np.append(LowArray,embedding[idx,:].reshape(1,-1),0)
        LowGen = np.append(LowGen,Gen[idx,:].reshape(1,-1),0)
        LowPhen.append(HeightPhen[idx])
        LowIndex.append(Iterator)
    if Data > Partitions[1] and Data <= Partitions[2]:
        InArray = np.append(InArray,embedding[idx,:].reshape(1,-1),0)
        InGen = np.append(InGen,Gen[idx,:].reshape(1,-1),0)
        InPhen.append(HeightPhen[idx])
        InIndex.append(Iterator)
    if Data > Partitions[2] and Data <= Partitions[3]:
        HiArray = np.append(HiArray,embedding[idx,:].reshape(1,-1),0)
        HiGen = np.append(HiGen,Gen[idx,:].reshape(1,-1),0)
        HiPhen.append(HeightPhen[idx])
        HiIndex.append(Iterator)
    Iterator += 0

# Delete the first row of each matrix that are populated with zeros.
LowArray = np.delete(LowArray,0,0)
InArray = np.delete(InArray,0,0)
HiArray = np.delete(HiArray,0,0)
LowGen = np.delete(LowGen,0,0)
InGen = np.delete(InGen,0,0)
HiGen = np.delete(HiGen,0,0)    
  
# %% Part 3a) GA #2 Analysis: Distance Metrics

# I will be using three distance metrics to validate the GA's results against
# individuals in the SoyNAM dataset. Using all 4400 SNPs, I will compare
# bin by bin with cosine similarity and hamming distance. After UMAP
# transformation, I will use euclidean distance in 2-D to compare. Comparison
# after UMAP is likely less reliable, but it is the easiest to visualize.

# Initialize Arrays to hold distance metrics.
LowMetrics = np.zeros((len(LowArray),len(BigArray),2))
InMetrics = np.zeros((len(InArray),len(BigArray),2))
HiMetrics = np.zeros((len(HiArray),len(BigArray),2))
Metrics = np.zeros((len(BigArray),2))

# There is definitely a more efficient way to do this than having three nested
# loops, but for organization's sake I'm keeping it separate.
# Calculate the Hamming distance between each GA-generated individual and
# every sample in the population. The goal here is to identify which bin
# contains the individual with a genotype that is closest to a GA-generated
# individual. This accomplishes the following:
    # 1. Validates that the GA is following the SNP constraint.
    # 2. Validates that the GA is producing logical genotypes.
         # i.e. similar phenotypes likely have similar genotypes.
         # do GA-generated individuals have genotypes close to their height bin?
    # 3. Shows the necessary number of SNPs to create a desired individual.
for idx, LowData in enumerate(LowGen):
    
    for idy, GA in enumerate(BigArray):
        
        LowMetrics[idx,idy,0] = scipy.spatial.distance.cosine(LowData,GA)
        LowMetrics[idx,idy,1] = scipy.spatial.distance.hamming(LowData,GA)
        
for idx, InData in enumerate(InGen):
    
    for idy, GA in enumerate(BigArray):
        
        InMetrics[idx,idy,0] = scipy.spatial.distance.cosine(InData,GA)
        InMetrics[idx,idy,1] = scipy.spatial.distance.hamming(InData,GA)

for idx, HiData in enumerate(HiGen):
    
    for idy, GA in enumerate(BigArray):
        
        HiMetrics[idx,idy,0] = scipy.spatial.distance.cosine(HiData,GA)
        HiMetrics[idx,idy,1] = scipy.spatial.distance.hamming(HiData,GA)
  
# %% Part 3b) Redefine reducer object and train on the Gen data and new GA individuals.

# Define reducer object. The parameters used have been optimized by a genetic 
# algorithm. A random forest was used to predict genotype from phenotype using
# the dimensionally reduced data. The UMAP parameters that produced the highest
# RF accuracy are used here.
reducer = umap.UMAP(n_neighbors=92,metric="cosine",
                    n_epochs=100,min_dist=0.075,learning_rate=1.36,
                    local_connectivity=5,repulsion_strength=2.76,
                    negative_sample_rate=24)

# Fit all genotype data AND individuals generated by the genetic algorithm
# to a 2-D space.
GAEmbedding = reducer.fit_transform(GenGa)

# Extract only the GA generated individuals. The "best" individual is extracted
# separately from the others.
EmbedGAPoints = GAEmbedding[-ProblemStatementIter-1:-1,:]
OriginalEmbed = GAEmbedding[:-ProblemStatementIter]

# %% Part 3b) Plot the Best Individuals in 2-D.

# Plot the embedded points together, with the GA-generated individuals
# accentuated.
norm = mpl.colors.Normalize(min(HeightPhen), max(HeightPhen))

plt.scatter(EmbedGAPoints[:,0],EmbedGAPoints[:,1],
            s=45,c='gold',edgecolors='black',marker='^',zorder=1000)
plt.scatter(GAEmbedding[OriginalID,0],GAEmbedding[OriginalID,1],
            s=150,c='orange',edgecolors='black',marker='*',zorder=100)
plt.scatter(OriginalEmbed[:,0],OriginalEmbed[:,1], norm=norm,
            s=1,c=HeightPhen,cmap='viridis',zorder=0)

plt.rcParams['figure.dpi'] = 300
plt.colorbar(label='Height (cm)')

plt.xlabel('UMAP Coordinate 1')
plt.ylabel('UMAP Coordinate 2')
plt.legend({'GA-Generated':'gold','Target':'orange'})

# Boost the quality of the plotted figure.
plt.show()
plt.savefig("betterquality.jpg", dpi=1200)

# # %% Part 3b) Plot the Best Individuals versus each bin.
# # For this visualization we hope to see overlap only within the bin of
# # close phenotype colors.

# # Create four subplots. The axis object will be used to assign data to each.
# fig, ax = plt.subplots(2,2,figsize=(10,10))

# # Create objects that can be used to assign a normalized colormap to
# # each of the subplots.
# cmap = mpl.cm.plasma
# norm = mpl.colors.Normalize(min(HeightPhen), max(HeightPhen))
# # norm = mpl.colors.BoundaryNorm(bounds,cmap.N)

# # The following code performs poorly when attempting to use a more iterative
# # method. As a result, I've hard-coded this visualization section. The
# # following three axes plot:
#     # "Low" - Genotypes from minimum phenotype value to bottom of target bin.
#     # "In" - Genotypes whose phenotype falls within the target bin.
#     # "Hi" - Genotypes from the top of target bin to the maximum phenotype.

# # For height data, the graph titles will automatically update if the target
# # value or bin sizes change.      
# ax[0][0].scatter(LowArray[:,0],LowArray[:,1],s=2,c=LowPhen,cmap='plasma', norm=norm)
# ax[0][0].scatter(EmbedPoints[:,0],EmbedPoints[:,1],
#             s=1.5,c='Green')
# ax[0][0].scatter(BestEmbed[:,0],BestEmbed[:,1],s=3,c='goldenrod')
# ax[0][0].set_title(str(round(Partitions[0])) + " cm to " + str(round(Partitions[1])) + " cm")

# ax[0][1].scatter(InArray[:,0],InArray[:,1],s=2,c=InPhen,cmap='plasma', norm=norm)
# ax[0][1].scatter(EmbedPoints[:,0],EmbedPoints[:,1],
#             s=1.5,c='Green')
# ax[0][1].scatter(BestEmbed[:,0],BestEmbed[:,1],s=3,c='goldenrod')
# ax[0][1].set_title(str(round(Partitions[1])) + " cm to " + str(round(Partitions[2])) + " cm")

# ax[1][0].scatter(HiArray[:,0],HiArray[:,1],s=2,c=HiPhen,cmap='plasma', norm=norm)
# ax[1][0].scatter(EmbedPoints[:,0],EmbedPoints[:,1],
#             s=1.5,c='Green')
# ax[1][0].scatter(BestEmbed[:,0],BestEmbed[:,1],s=3,c='goldenrod')
# ax[1][0].set_title(str(round(Partitions[2])) + " cm to " + str(round(Partitions[3])) + " cm")


# # Final Plot: this plot displays all genotype points colored by their heights.
# # However, the GA-generated individuals are excluded, except for the best.
# ax[1][1].scatter(embedding[:,0],embedding[:,1],s=2,c=HeightPhen,cmap='plasma', norm=norm)
# # ax[1][1].scatter(EmbedPoints[:,0],EmbedPoints[:,1],
# #             s=1.5,c='Green')
# ax[1][1].scatter(BestEmbed[:,0],BestEmbed[:,1],s=3,c='goldenrod')
# ax[1][1].set_title("All Heights")

# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax[:][1])
# cbar.ax.set_title("Height(cm)")

# fig
# plt.savefig("Binning.jpg", dpi=2000)

# %% Part 3c) Plot Generated Individuals versus target cluster.

# Initialize an Array for visualization in UMAP later.
TargetNeighborsEmbed = np.zeros((1,2)) 

TargetNeighborID = WhatNeighbor[OriginalID]

for Neighbor in TargetNeighborID:
    TargetNeighborsEmbed = np.append(TargetNeighborsEmbed,embedding[Neighbor,:].reshape(1,-1),0)

TargetNeighborsEmbed = np.delete(TargetNeighborsEmbed,0,0)

# Plot the embedded points together, with the GA-generated individuals
# accentuated.
# plt.scatter(EmbedPoints[:,0],EmbedPoints[:,1],
#             s=5,c='Lime')
plt.scatter(GAEmbedding[:,0],GAEmbedding[:,1],
            s=1,c=HeightProtein[:,0],cmap='plasma')
plt.scatter(TargetNeighborsEmbed[:,0],
            TargetNeighborsEmbed[:,1], s=7.5, c="turquoise"
            )


plt.rcParams['figure.dpi'] = 300
plt.colorbar()

# Boost the quality of the plotted figure.
plt.show()
plt.savefig("TargetClusterGraph.jpg", dpi=1200)


# %% Part 4a) GA #3: Initial Definitions, Neighbor Population Creation

global NumGen3
global PopSize3
global IncPop

# Bring in phenotype data for all relevant characteristics.
HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
# Import the genotype (SNP Array).
Gen = GenImport()

# This GA has an early stop condition, so I am giving it a high number of
# generations so it can hit that threshold.
NumGen3 = 20000
PopSize = 20

IncPop = np.zeros((1,4400))
Target = 2422
IncPopIndexes = WhatNeighbor[Target]
IncPopPhen = np.zeros((1,2))

for Index in IncPopIndexes:
    IncPop = np.append(IncPop,Gen[Index,:].reshape(1,-1),axis=0)
    IncPopPhen = np.append(IncPopPhen,HeightProtein[Index,:].reshape(1,-1),axis=0)


IncPop = np.delete(IncPop,0,0)
DesiredValue = regressor.predict(Gen[Target,:].reshape(1,-1))
DeletedGen = Gen[Target,:]

Gen = np.delete(Gen,Target,0)
HeightProtein = np.delete(HeightProtein,Target,0)

InitPop = IncPop
# Use this line of code as many times as necessary to make InitPop longer than
# the population size.
InitPop = np.append(InitPop,IncPop,0)

# %% Part 4a) GA #3: Fitness Function for Inverse Genotype Design.

# This fitness function is extremely similar to fitness_function2, but
# has been changed to only replace points that exceed the 20 SNP limit from
# within the known neighbor cluster of the target genotype.

def fitness_function3(sol,sol_idx):

    global ga_instance2
    global ga_instance3
    global PopSize    
    global Redundancy  
    # Preallocate a vector with the same length as the number of genotypes
    # available in the data.
    DistanceHold = np.zeros((len(IncPop[:,0]),1))
    # Calculate the Hamming distance (proportion of mismatches) between 
    # the current GA solution and every other point in the data.
    for idx, TestPoint in enumerate(IncPop):
        DistanceHold[idx,:] = scipy.spatial.distance.hamming(TestPoint,sol)
    
    # If there are more SNPs in the current solution than the set tolerance
    # allows, throw out the solution by replacing it with a genotype from the
    # dataset.
    if min(DistanceHold) > SNPTol/len(IncPop[0,:]):
        if ga_instance3.generations_completed >= NumGen3 - 1:
            pass
        else:    
            # Different from last fitness function. The GA is now anchored only
            # to data points within the cluster of neighbors defined. 
            RandPop = random.randrange(len(InitPop))
            ga_instance3.population[sol_idx,:] = InitPop[RandPop,:]
    # If the current solution is a genotype that exists in the dataset, set
    # its fitness to zero. This ensures that it will not be in the next
    # generation's population. However, because dataset genotypes introduced by
    # the "if" statement will exist in the population for one generation, their
    # genotypes will still have the chance to mate and mutate! Win/win.
    elif min(DistanceHold) == 0:
        return 0
        
    # Use the random forest to predict the phenotype of the each member of
    # the population.
    Predict = regressor.predict(sol.reshape(1,-1))    
    # Would it be better to use MAPE or MSE here?
    # However, the GA package is coded to find the HIGHEST number for fitness, 
    # so I must take the inverse of MAPE, as lower MAPE is actually better.
    # fitness = 1/MAPE(DesiredValue,Predict)
    if np.sum(abs(DesiredValue-Predict)) == 0:
        fitness = 100000
    else:
        fitness = 1/(np.sum(abs(DesiredValue-Predict)))
    return fitness

# %% Part 4a) GA #3: Definitions of Verbose Functions

# global Redundancy  
Redundancy = 0

# The GA will automatically print all of the following information after each 
# generation.
def on_generation3(ga_instance):
    global Redundancy
    global StartTime
    global TotalTime
    global GenCount
    
    if max(ga_instance.last_generation_fitness) == max(ga_instance.previous_generation_fitness):
        Redundancy += 1
    else:
        Redundancy = 0
        
    if ga_instance.generations_completed == 0:
        pass
    else:
        if Redundancy >= 300 and max(ga_instance.last_generation_fitness) <= 100000:
            GenCount += ga_instance.generations_completed
            return "stop"
            
    if max(ga_instance.last_generation_fitness) == 100000:
        GenCount += ga_instance.generations_completed
        return "stop"
    
    print("Fitness of Best Solution this Iter: ", ga_instance.best_solution()[1])
    print("Best Overall Fitness: ", AllTimeBest)
    print("SNPs of best Fitness: ", BestIndivSNPs)
    print("Lowest SNPs Achieved: ", LowestSNPs)
    print("Time Elapsed: ", (time.time()-StartTime))
    print("Total Time Elapsed: ", (time.time()-TotalTime))
    # print("Population:")
    # print(ga_instance.population)
    print("Target", TargetCount+1, "of 19.")
    print("Iterations for this Target: ", (Iter+1))
    print("Generation: ", ga_instance.generations_completed)
    print("Redundancy: ", Redundancy)
    print("________________________________________________")

# %% Part 4a) GA #3: Incremental Optimization Proof of Concept, this is only
# for a single run!!! Use Part 4b) to reproduce results.

global ga_instance3
    
ga_instance3 = pygad.GA(# How many GA iterations?
                       num_generations=NumGen3,
                       # Each generation how many individuals will cross over?
                       num_parents_mating=5,
                       # See above for my fitness function.
                       fitness_func=fitness_function3,
                       # Population Size of the GA: how many parameter sets?
                       sol_per_pop=PopSize,
                       # I'm optimizing 4400 SNPs, to create an individual
                       # with desired phenotype.
                       num_genes=NumGenes,
                       # Anchor the GA at existing points to ensure valid
                       # results and speed up the process.
                       initial_population = np.random.permutation(InitPop)[:PopSize,:],
                       # Data types of each parameter. All should be Int 
                       # (one-hot encoding)
                       gene_type = int,
                       # GA randomly picks initial values. 
                       # These are the allowed range.
                       init_range_low = 0,
                       init_range_high = 2,
                       # How many individuals kept unchanged between generations.
                       keep_elitism = 2,
                       crossover_probability = 0.3,
                       # In two solutions, two points are selected, 
                       # between which all points are switched.
                       crossover_type = "two_points",
                       # I've lowered this probability, because the search space
                       # is relatively small.
                       mutation_probability = 0.002,
                       
                       # See above for definitions of these:
                       gene_space=gene_space2,
                       on_generation=on_generation3,
                       on_start=on_start3,
                       
                       
                       # GA automatically saves an array of the best solution 
                       # every generation. Makes it easier to visualize the best
                       # solution later.
                       save_best_solutions=False,
                       save_solutions=False
                       )

# %% Part 4a) GA #3: Run!     

# Initialize and global the current time, so the runtime of the GA can be
# tracked.
# global StartTime
StartTime = time.time()

ga_instance3.run()

# Pull information about the final population of solutions for visualization
# and analysis.
AllFitness = ga_instance3.last_generation_fitness
BestIndividual = ga_instance3.best_solutions[0]
LastSolutions = np.array(ga_instance3.solutions[-PopSize-1:-1])

# %% Part 4a) Extract the Best Individuals from the Population.

BigArray = np.zeros((1,4400))
Fitness = []

# Only solutions that have phenotypes within 0.05% of the target value are
# added to a master array of solutions. Their fitnesses are also recorded.
for idx, fitness in enumerate(AllFitness):
    if fitness >= 0.5:
        BigArray = np.append(BigArray,LastSolutions[idx].reshape(1,-1),0)
        Fitness.append(fitness)
        
# Delete the holding row used to initialize the array.
# Also, flip the array such that the best individual is at index -1.       
BigArray = np.delete(BigArray,0,0)
NumIndividuals = len(BigArray)

# Compile all data points together in order!
GenGa = np.append(Gen,BigArray,axis=0)

# %% Part 4b) GA #3: Testing all Target Values with Iteration.
# This GA will loop until reaching a stagnant fitness (local optimum),
# and iterate a few times to see if each target genotype can be reached from
# its neighbors.

# global Iterations
# global GenCount
# global Iter
# global TotalTime 
# global ga_instance3
# global TargetCount
# How many times to run each target.
Iterations = 10
# Amongst all iterations for a given target, what is the best fitness?
AllTimeBest = 0
# What is the lowest number of SNPs between the target and a generated individual?
LowestSNPs = 100
# How many SNPs are required to reach the individual with AllTimeBest fitness?
BestIndivSNPs = 0

# Initialize lists and counters.
LowestSNPPerTarget = []
BestFitnessSNPPerTarget = []
TotalGenerationsRequired = []
IterationsRequired = []
TargetCount = 0

# Organize the cluster of 19 neighbors together into a list.
AllTargets = [1215]
for Neighbor in WhatNeighbor[1215]:
    AllTargets.append(Neighbor)
    
# Keep track of the global time that has passed since the loop started.    
TotalTime = time.time()
    
for idx, Target in enumerate(AllTargets):
    
    # Reset all the tracker values for each target.
    AllTimeBest = 0
    LowestSNPs = 100
    BestIndivSNPs = 0
    # Keep track of how many generations have been completed within GA.
    GenCount = 0
    IterCount = 0
    
    # if AllTimeBest >= 100000: 
    #     print("Found.")
    #     break
    
    # global DesiredValue
    
    # Reimport the genotype and phenotype data, restore any previous deletion.
    Gen = GenImport()
    HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
    
    # Find what value the random forest assigns to our target.
    DesiredValue = regressor.predict(Gen[Target,:].reshape(1,-1))
    
    # Define the included population, every member of the cluster but target.
    IncPop = np.zeros((1,4400))
    IncPopIndexes = WhatNeighbor[Target]
    IncPopPhen = np.zeros((1,2))

    # Fill the genotypes and phenotypes of the cluster members into their arrays.
    for Index in IncPopIndexes:
        IncPop = np.append(IncPop,Gen[Index,:].reshape(1,-1),axis=0)
        IncPopPhen = np.append(IncPopPhen,HeightProtein[Index,:].reshape(1,-1),axis=0)

    # Remove the placeholder rows.
    IncPop = np.delete(IncPop,0,0)
    IncPopPhen = np.delete(IncPopPhen,0,0)
    
    # Save the genotype 
    DeletedGen = Gen[Target,:]
    Gen = np.delete(Gen,Target,0)
    HeightProtein = np.delete(HeightProtein,Target,0)

    # Create copies of the neighbor population using this code until
    # the initial population is larger than "PopSize".
    InitPop = IncPop
    InitPop = np.append(InitPop,IncPop,0)
    
    for Iter in range(Iterations):
        
        ga_instance3 = pygad.GA(# How many GA iterations?
                               num_generations=NumGen3,
                               # Each generation how many individuals will cross over?
                               num_parents_mating=5,
                               # See above for my fitness function.
                               fitness_func=fitness_function3,
                               # Population Size of the GA: how many parameter sets?
                               sol_per_pop=PopSize,
                               # I'm optimizing 4400 SNPs, to create an individual
                               # with desired phenotype.
                               num_genes=4400,
                               # Anchor the GA at existing points to ensure valid
                               # results and speed up the process.
                               initial_population = np.random.permutation(InitPop)[:PopSize,:],
                               # Data types of each parameter. All should be Int 
                               # (one-hot encoding)
                               gene_type = int,
                               # GA randomly picks initial values. 
                               # These are the allowed range.
                               init_range_low = 0,
                               init_range_high = 2,
                               # How many individuals kept unchanged between generations.
                               keep_elitism = 2,
                               crossover_probability = 0.3,
                               # In two solutions, two points are selected, 
                               # between which all points are switched.
                               crossover_type = "two_points",
                               # I've lowered this probability, because the search space
                               # is relatively small.
                               mutation_probability = 0.002,
                               
                               # See above for definitions of these:
                               gene_space=gene_space2,
                               on_generation=on_generation3,
                               on_start=on_start3,
                               
                               save_best_solutions=True)
            
        # global StartTime
        StartTime = time.time()
        ga_instance3.run()
    
        # Save the best fitness and the genotype that produced it.
        # In addition, calculate the hamming distance of this individual
        # from the target.
        BestFitness = max(ga_instance3.last_generation_fitness)
        BestIndividual = ga_instance3.best_solutions[0]
        Hamming = scipy.spatial.distance.hamming(BestIndividual,DeletedGen)*4400
        
        # Update AllTimeBest if a new record is set.
        if BestFitness > AllTimeBest:
            AllTimeBest = BestFitness
            BestIndivSNPs = math.floor(Hamming)
        
        # If the best individual of any run has the lowest SNP count, update.
        if Hamming < LowestSNPs:
            LowestSNPs = math.floor(Hamming)
        
        IterCount += 1
        
        # 100000 is used as a marker for hitting the target. 
        # This means that we have found the exact target,
        # this automatically assigns a SNP value of 0 for a direct hit.
        if AllTimeBest == 100000:
            LowestSNPPerTarget.append(0)
            BestFitnessSNPPerTarget.append(0)
            # print("Found.")
            break
        
    # If the exact target has not been found, the above code will not run,
    # and therefore this if statement will be true. Otherwise, a double insert
    # will occur.    
    if len(LowestSNPPerTarget) == TargetCount:
        LowestSNPPerTarget.append(LowestSNPs)
        BestFitnessSNPPerTarget.append(BestIndivSNPs)
        
    TotalGenerationsRequired.append(GenCount)
    IterationsRequired.append(IterCount)
        
    TargetCount += 1

# %% Part 5) Add More Phenotypes, Perform Case Study

""" Switch to using Height, Protein, and Size. This section will be an in silico
case study examining inverse phenotype design in soybeans. Of great interest to
the soybean industry (and the poultry industry it supports), is a higher protein
content for the poultry animals that soybeans are grown to feed. Specifically,
a higher protein density within the whole plant would be preferable.
The fitness of individuals in this case study will be determined as a trade-off
between their protein content and their height and seed size. Solutions will be
rewarded for decreasing height and seed size while increasing protein content.
Any height and seed size that exceeds the population mean for their respective
phenotype will be aggressively punished. 

Due to the massive search space of this problem and as shown in the code above,
this case study will simulate an incremental optimization approach. This paradigm
is also known as "Design, Build, Test, Learn". Since this is predominantly an
in silico approach, Design, Build, and Test will be performed simulataneously
and the random forest prediction for the solution will be taken as correct. 
Due to low prediction errors as above, this is an acceptable proof of concept.

Individuals generated at each iteration of the GA will be added to the population
and the random forest will be retrained on these individuals. A new iteration
will then push the boundaries of protein content.


For now, I plan to perform 5 runs of the GA per iteration of DBTL, adding the
best individuals into the population. Then, up to five iterations of the DBTL
itself.
"""

# %% Part 5a) RF: Regression of Phenotype data.

# Bring in phenotype data for all relevant characteristics.
HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
# Import the genotype (SNP Array).
Gen = GenImport()

# Delete the phenotype outlier from analysis.
HeightProtein = np.delete(HeightProtein,2459,axis=0)
HeightProteinSize = np.delete(HeightProteinSize,2459,axis=0)
Gen = np.delete(Gen,2459,axis=0)

# Code mostly copied from above. For the proof of concept, include all data in RF.
# Accuracy will be assessed by K-Fold Cross validation. Here, we calculate the
# "in-bag" score by fitting all data and finding the accuracy of random fraction.
x_train, x_test, y_train, y_test = train_test_split(Gen,HeightProteinSize,test_size=0.2)

# Initialize the RF object. For this project we use default parameters because
# tuning gave marginal performance boosts at best.
HPSregressor = RandomForestRegressor()

# If using a GA, use all of the data to train RF:
    # Sample weight can be included by running the below sections and removing
    # the commented section below.
HPSregressor.fit(Gen,HeightProteinSize,#sample_weight=sample_weight
              )

# Predict the held-out genotype/phenotype data. If x_train/y_train used to
# train regressor, this is true accuracy. If x_test is represented in training
# set, this is in-bag accuracy.
HPSPredict = HPSregressor.predict(x_test)

HPSFullPredict = HPSregressor.predict(Gen)

# Functions for assessing the accuracy of the RF.
# Mean Squared Error and Mean Average Percentage Error.
HPSMSE = mean_squared_error(y_test,HPSPredict)
HPSMeanAPE = MAPE(y_test,HPSPredict)


# %% Part 5a) K-Fold Cross Validation with 3 Phenotypes

# How many sections should the data be split into?
Folds = 5

# This object will split any dataset fed to its .split() method into n_splits
# parts by index.
kf = KFold(n_splits=Folds, random_state=None)
# Accuracy scores will be stored in this array.
MAPEFold = []
# Initializing this variable to make KFold verbose.
FoldCount = 0

for train_index, test_index in kf.split(Gen):
    # Partition the data using the indexes provided by KFold.split(). 
    # IMPORTANT: decide whether to use sample weights for the whole K-Fold.
    x_train, x_test = Gen[train_index,:], Gen[test_index,:]
    y_train, y_test = HeightProteinSize[train_index], HeightProteinSize[test_index]
    # sample_test = sample_weight[train_index]

    # Create a fresh instance of the regressor each iteration, and train it.
    FoldRegressor = RandomForestRegressor()    
    FoldRegressor.fit(x_train,y_train,# sample_weight=sample_test
                      )

    # Predict the phenotypes of all the withheld genotype information.
    # Then, calculate the MAPE as an accuracy measurement.
    FoldPredict = FoldRegressor.predict(x_test)
    MAPEFold.append(MAPE(y_test,FoldPredict))
    
    # Verbose section. How many folds have been completed?
    FoldCount += 1
    print("Fold Completed: {}".format(FoldCount))

# Calculate and print the average MAPE during KFold.
AvgMAPE = sum(MAPEFold)/Folds
print("Average MAPE: ", AvgMAPE)

# %% Part 5a) Constraint Definitions, How to Define Fitness?

global HeightMean, SeedMean, ProteinMean, ProteinStDev

# Find the population mean of Height and Seed Size.
HeightMean = np.mean(HeightProteinSize[:,0])
ProteinMean = np.mean(HeightProteinSize[:,1])
SeedMean = np.mean(HeightProteinSize[:,2])
ProteinMax = max(HeightProteinSize[:,1])
ProStDev = statistics.stdev(HeightProteinSize[:,1])

HeightRange = max(HeightProteinSize[:,0]) - min(HeightProteinSize[:,0])
ProteinRange = max(HeightProteinSize[:,1]) - min(HeightProteinSize[:,1])
SeedRange = max(HeightProteinSize[:,2]) - min(HeightProteinSize[:,2])

def HeightFitness(SolHeight):
    
    if SolHeight <= HeightMean:
        HeightFit = abs(HeightMean-SolHeight)*0.005
    else:
        HeightFit = abs(HeightMean-SolHeight)*(-1)
    
    return HeightFit
    
def SeedFitness(SolSeed):
    
    if SolSeed <= SeedMean:
        SeedFit = abs(SeedMean-SolSeed)*0.01
    else:
        SeedFit = abs(SeedMean-SolSeed)*(-5)
    
    return SeedFit

def ProteinFitness(SolProtein):
    
    ProteinThreshold = ProteinMean + ProStDev
    Protein95 = ProteinMean + ProStDev*2
    
    if SolProtein <= ProteinMean:
        ProteinFit = abs(ProteinMean-SolProtein)*(-5)
    elif SolProtein > ProteinMean and SolProtein < ProteinThreshold:
        ProteinFit = abs(ProteinMean-SolProtein)*0.5
    elif SolProtein >= ProteinThreshold and SolProtein < Protein95:
        ProteinFit = ProStDev*0.5 + abs(SolProtein-ProteinThreshold)*10
    else:
        ProteinFit = ProStDev*0.5 + ProStDev*10 + abs(SolProtein-Protein95)*100
    return ProteinFit
    

# %% Part 5b) Incremental Optimization Case Study
# %% Part 5b) GA #4: Initial Definitions

# global SNPTol

# How many full DBTL cycles will run?
DBTLTotal = 20
# How many runs to find good individuals will occur?
RunTotal = 5

# Same as above. Small population is better to allow for less distance
# calculations in fitness function.
PopSize = 20 
# After 10 generations of stagnancy, terminate the GA.
# Avoids domination of the population by a single strong individual.
Redundancy4 = 10
# Give the GA enough time to find good individuals but not enough to 
# dominate the entire population. While it is still acceptable to terminate
# by the redundancy, it is less preferable.
NumGen4 = 50

# Allowed values in the GA solutions.
gene_space4 = [0, 1, 2]

sample_weight = np.ones((len(Gen[:,0]),1))

SNPTol = 10

# Bring in phenotype data for all relevant characteristics.
HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
# Import the genotype (SNP Array).
GenCaseStudy = GenImport()



# %% Part 5b) GA #4: Fitness Function

# This fitness function will be more complex than the others above because
# we are now performing an optimization with multiple constraints.
# These constraints used the penalty method, in which a steep penalty is
# applied to any heights or seed sizes that exceed the population mean.

def fitness_function4(sol,sol_idx):

    global ga_instance4
    global PopSize
    global Redundancy4 
    # Preallocate a vector with the same length as the number of genotypes
    # available in the data.
    DistanceHold = np.zeros((len(GenCaseStudy[:,0]),1))
    # Calculate the Hamming distance (proportion of mismatches) between 
    # the current GA solution and every other point in the data.
    for idx, TestPoint in enumerate(GenCaseStudy):
        DistanceHold[idx,:] = scipy.spatial.distance.hamming(TestPoint,sol)
    
    # If there are more SNPs in the current solution than the set tolerance
    # allows, throw out the solution by replacing it with a genotype from the
    # dataset.
    if min(DistanceHold) > SNPTol/len(GenCaseStudy[0,:]):
        if ga_instance4.generations_completed >= NumGen4 - 1:
            pass
        else:    
            # Different from last fitness function. The GA is now anchored only
            # to data points within the cluster of neighbors defined. 
            RandPop = random.randrange(len(GenCaseStudy))
            ga_instance4.population[sol_idx,:] = GenCaseStudy[RandPop,:]
    # If the current solution is a genotype that exists in the dataset, set
    # its fitness to zero. This ensures that it will not be in the next
    # generation's population. However, because dataset genotypes introduced by
    # the "if" statement will exist in the population for one generation, their
    # genotypes will still have the chance to mate and mutate! Win/win.
    elif min(DistanceHold) == 0:
        return 0
        
    # Use the random forest to predict the phenotype of the each member of
    # the population.
    HPSPredictBig = HPSregressor.predict(sol.reshape(1,-1))    
    HPSPredict = HPSPredictBig[0]
    # Call to pre-written fitness functions.
    HeightFit = HeightFitness(HPSPredict[0])
    ProteinFit = ProteinFitness(HPSPredict[1])
    SeedFit = SeedFitness(HPSPredict[2])
    
    # Combine all fitnesses (penalties, rewards, etc.).
    fitness = HeightFit + ProteinFit + SeedFit
    
    return fitness

# %% Step 5b) GA #4: Verbose Definitions

# global Redundancy
Redundancy = 0 

# The GA will automatically print all of the following information after each 
# generation.
def on_generation4(ga_instance):
    global Redundancy
    global StartTime
    global DBTLTime
    global TotalTime
    
    if max(ga_instance.last_generation_fitness) == max(ga_instance.previous_generation_fitness):
        Redundancy += 1
    else:
        Redundancy = 0
        
    if ga_instance4.generations_completed == 0:
        pass
    else:
        if Redundancy >= Redundancy4:
            return "stop"
    

    # print("Best Overall Fitness: ", AllTimeBest)
    print("Time Elapsed: ", (time.time()-StartTime))
    print("Time Elapsed this DBTL: ", (time.time()-DBTLTime))
    print("Total Time Elapsed: ", (time.time()-TotalTime))

    BestFitBig = HPSregressor.predict(ga_instance.best_solution()[0].reshape(1,-1))
    BestFit = BestFitBig[0]

    print("Fitness of Best Solution this Run: ", ga_instance.best_solution()[1])
    print("Best Sol: Protein: ", BestFit[1], ", Height: ", BestFit[0], 
          " cm, Seed Size: ", BestFit[2], " g/100 seeds.")

    print("Runs this DBTL: ", (GAIter+1))
    print("DBTL Iteration: ", (DBTLCycle+1))
    print("Generation: ", ga_instance.generations_completed)
    print("Redundancy: ", Redundancy)
    print("________________________________________________")
    
    
def on_start4(ga_instance):
    print("Runs for this Target: ", (GAIter+1))
    print("DBTL Iteration: ", (DBTLCycle+1))

    print("Generation Time Elapsed: ", (time.time()-StartTime))
    print("Time Elapsed this DBTL: ", (time.time()-DBTLTime))
    print("Total Time Elapsed: ", (time.time()-TotalTime))
    print("________________________________________________")        

# %% Part 5b) GA #4: Master Loop

# This section will become a doubly nested for loop that will run full
# iterations of the DBTL algorithm.

global ga_instance4
TotalTime = time.time() 

for DBTLCycle in range(DBTLTotal):
    
    sample_weight_adder = 10
    
    DBTLTime = time.time()
    
    # Individuals that will be added to the population between generations
    # are stored here.
    GenAddHold = np.zeros((1,4400))
    
    # The phenotypes of best individuals to be added are included here.
    PhenAddHold = np.zeros((1,3))

    # Retrain the random forest classifier anytime new information is included.
    if DBTLCycle != 0:
        print("Retraining Random Forest for DBTL Cycle: ", DBTLCycle)
        print("________________________________________________")
        HPSregressor = RandomForestRegressor()
        HPSregressor.fit(GenCaseStudy,HeightProteinSize,#sample_weight=sample_weight
                         )
    
    for GAIter in range(RunTotal):
        
        ga_instance4 = pygad.GA(# How many GA iterations?
                               num_generations=NumGen4,
                               # Each generation how many individuals will cross over?
                               num_parents_mating=5,
                               # See above for my fitness function.
                               fitness_func=fitness_function4,
                               # Population Size of the GA: how many parameter sets?
                               sol_per_pop=PopSize,
                               # I'm optimizing 4400 SNPs, to create an individual
                               # with desired phenotype.
                               num_genes=4400,
                               # Anchor the GA at existing points to ensure valid
                               # results and speed up the process.
                               initial_population = np.random.permutation(GenCaseStudy)[:PopSize,:],
                               # Data types of each parameter. All should be Int 
                               # (one-hot encoding)
                               gene_type = int,
                               # GA randomly picks initial values. 
                               # These are the allowed range.
                               init_range_low = 0,
                               init_range_high = 2,
                               # How many individuals kept unchanged between generations.
                               keep_elitism = 2,
                               crossover_probability = 0.3,
                               # In two solutions, two points are selected, 
                               # between which all points are switched.
                               crossover_type = "two_points",
                               # I've lowered this probability, because the search space
                               # is relatively small.
                               mutation_probability = 0.002,
                               
                               # See above for definitions of these:
                               gene_space=gene_space4,
                               on_generation=on_generation4,
                               on_start=on_start4,
                               
                               
                               # GA automatically saves an array of the best solution 
                               # every generation. Makes it easier to visualize the best
                               # solution later.
                               save_best_solutions=True,
                               save_solutions=False
                               )
        
        StartTime = time.time()

        ga_instance4.run()    
            
        BestFitness = max(ga_instance4.last_generation_fitness)
        BestIndividual = ga_instance4.best_solutions[-1]
        BestIndividualPhen = HPSregressor.predict(BestIndividual.reshape(1,-1))
        
        GenAddHold = np.append(GenAddHold,BestIndividual.reshape(1,-1),0)
        PhenAddHold = np.append(PhenAddHold,BestIndividualPhen.reshape(1,-1),0)
        sample_weight = np.append(sample_weight,
                                  np.array(sample_weight_adder).reshape(1,-1),0)
    
    GenAddHold = np.delete(GenAddHold,0,0)
    PhenAddHold = np.delete(PhenAddHold,0,0)
    
    GenCaseStudy = np.append(GenCaseStudy,GenAddHold,0)
    HeightProteinSize = np.append(HeightProteinSize,PhenAddHold,0)
    
# %% Part 5c) Histogram of Protein Contents

# Bring in phenotype data for all relevant characteristics.
HeightPhen, HeightProtein, HeightProteinSize, Phen = PhenImport()
# Import the genotype (SNP Array).
Gen = GenImport()

# Delete the phenotype outlier from analysis.
HeightProtein = np.delete(HeightProtein,2459,axis=0)
HeightProteinSize = np.delete(HeightProteinSize,2459,axis=0)
Gen = np.delete(Gen,2459,axis=0)

bins = np.linspace(math.floor(min(HeightProteinSize[:,1])),
                   math.ceil(max(HeightProteinSize[:,1])))

PhenMaster = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\PhenMaster.csv",
                 delimiter=",")

GenMaster = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\GenMaster.csv",
                 delimiter=",", dtype=int)

Original = HeightProteinSize[:,1]
GAGenerated = PhenMaster[:,1]
GenCaseStudy = np.append(Gen,GenMaster,axis=0)

plt.figure()
plt.hist([Original, GAGenerated], bins, stacked=True,
         color=['blue','gold'])
plt.xlabel('Protein Content (%)')
plt.ylabel('Number of Individuals')
plt.legend({"Original Data": 'blue', "GA-Generated": 'gold'})
plt.title('True Phenotype Distribution with GA Individuals')
plt.xlim([30.5, 37])
plt.ylim([0, 650])
plt.show()

# %% Part 5c) Plot histogram based on RF results as well.


plt.figure()
plt.hist([HPSFullPredict[:,1], GAGenerated], bins, stacked=True,
         color=['blue','gold'])
plt.xlabel('Protein Content (%)')
plt.ylabel('Number of Individuals')
plt.legend({"RF Predicted Original": 'blue', "GA-Generated": 'gold'})
plt.title('RF-Predicted Distribution with GA Individuals')
plt.xlim([30.5, 37])
plt.ylim([0, 650])
plt.show()


# %% Part 5c) Scatter Plots of Height/Seed Size vs. Protein

GAHeight, GAProtein, GASeed = PhenMaster[:,0], PhenMaster[:,1], PhenMaster[:,2]
OrigHeight, OrigProtein, OrigSeed = HeightProteinSize[:,0], HeightProteinSize[:,1], HeightProteinSize[:,2]
RFHeight, RFProtein, RFSeed = HPSFullPredict[:,0], HPSFullPredict[:,1], HPSFullPredict[:,2]

plt.figure()
plt.scatter(OrigHeight,OrigProtein,c="blue",s=10)
plt.scatter(GAHeight,GAProtein,c="gold",marker="^",s=35,edgecolors='black')
plt.xlabel('Height (cm)')
plt.ylabel('Protein Content (%)')
plt.legend({"Original Data": 'blue', "GA-Generated": 'gold'})
plt.xlim([60, 140])
plt.ylim([30, 38])
plt.show()

plt.figure()
plt.scatter(RFHeight,RFProtein,c="blue",s=10)
plt.scatter(GAHeight,GAProtein,c="gold",marker="^",s=35,edgecolors='black')
plt.xlabel('Height (cm)')
plt.ylabel('Protein Content (%)')
plt.legend({"Original Data": 'blue', "GA-Generated": 'gold'})
plt.xlim([60, 140])
plt.ylim([30, 38])
plt.show()

plt.figure()
plt.scatter(OrigSeed,OrigProtein,c="blue",s=10)
plt.scatter(GASeed,GAProtein,c="gold",marker="^",s=35,edgecolors='black')
plt.xlabel('Seed Size (g/100 seeds)')
plt.ylabel('Protein Content (%)')
plt.legend({"Original Data": 'blue', "GA-Generated": 'gold'})
plt.xlim([11, 22])
plt.ylim([30, 38])
plt.show()

plt.figure()
plt.scatter(RFSeed,RFProtein,c="blue",s=10)
plt.scatter(GASeed,GAProtein,c="gold",marker="^",s=35,edgecolors='black')
plt.xlabel('Seed Size (g/100 seeds)')
plt.ylabel('Protein Content (%)')
plt.legend({"Original Data": 'blue', "GA-Generated": 'gold'})
plt.xlim([11, 22])
plt.ylim([30, 38])
plt.show()


# %% Part 5c) Find Nearest Neighbor, Phenotype Difference of GA individuals:
    # How many SNPs are required?
    # Is the phenotype actually different from its nearest neighbor?
    
# Create an array to hold all information to be put in tables in manuscript.
# 5 individuals per DBTL, 20 DBTLs total. 7 columns for:
    #* Height
    #* Protein
    #* Size
    #* Parent Index
    #* Phenotype Difference
    #* RF Phenotype Difference
    #* SNP Difference from Parent
    
# Add the GA-phenotypes to the array for the following analysis. 
# HeightProteinSize = np.append(HeightProteinSize,PhenMaster,axis=0)
# HPSFullPredict = np.append(HPSFullPredict,PhenMaster,axis=0)
CaseStudyDataAnalysis = np.zeros((7,5,20))
CaseStudyFullList = np.zeros((7,100))
MiniCounter = 0
DBTLCounter = 0

for idx, GenCheck in enumerate(GenMaster):
    
    DistanceList = []
    
    for GenPop in GenCaseStudy:
        
        if scipy.spatial.distance.hamming(GenCheck,GenPop) == 0:
            DistanceList.append(100)
        else:
            DistanceList.append(scipy.spatial.distance.hamming(GenCheck,GenPop))

    CaseStudyDataAnalysis[:3,MiniCounter,DBTLCounter] = PhenMaster[idx,:]
    NeighborIdx = DistanceList.index(min(DistanceList))
    CaseStudyDataAnalysis[3,MiniCounter,DBTLCounter] = NeighborIdx
    
    CaseStudyDataAnalysis[4,MiniCounter,DBTLCounter] = PhenMaster[idx,1] - HeightProteinSize[NeighborIdx,1] 
    CaseStudyDataAnalysis[5,MiniCounter,DBTLCounter] = PhenMaster[idx,1] - HPSFullPredict[NeighborIdx,1] 
    
    CaseStudyDataAnalysis[6,MiniCounter,DBTLCounter] = min(DistanceList)*4400
    
    CaseStudyFullList[:3,idx] = PhenMaster[idx,:]
    CaseStudyFullList[3,idx] = NeighborIdx
    CaseStudyFullList[4,idx] = PhenMaster[idx,1] - HeightProteinSize[NeighborIdx,1]
    CaseStudyFullList[5,idx] = PhenMaster[idx,1] - HPSFullPredict[NeighborIdx,1] 
    CaseStudyFullList[6,idx] = min(DistanceList)*4400

    MiniCounter += 1
    
    if MiniCounter == 5:
        MiniCounter = 0
        DBTLCounter += 1
        print("DBTL Counter:", DBTLCounter)


AveragePhenChange = []
AverageRFPhenChange = []

for DBTL in range(20):
    AveragePhenChange.append(np.mean(CaseStudyDataAnalysis[4,:,DBTL]))
    AverageRFPhenChange.append(np.mean(CaseStudyDataAnalysis[5,:,DBTL]))
    
SmoothPhenChange = scipy.signal.savgol_filter(AverageRFPhenChange, 19, 3) 
   
# %% Part 5c) Plot Average Phenotype Change

plt.figure()
plt.plot(range(1,21),SmoothPhenChange,c='blue')
plt.scatter(range(1,21), AverageRFPhenChange,c='gold',marker='^',edgecolors='black')
plt.xticks(range(1,21))
plt.xlabel('DBTL Cycle')
plt.ylabel('Average Phenotype Difference from Parent')
plt.legend({'Savitsky-Golay smoothing': 'yellow'})
plt.show()