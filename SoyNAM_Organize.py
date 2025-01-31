# -*- coding: utf-8 -*-
"""
Created by: Joe Zavorskas
Start Date: 10/25/2022
Last Edit: 10/28/2022

The purpose of this file is to experiment with training
an autoencoder to compress available SNP data so
that inferences can be made from the compressed space to
the phenotype.

"""

# %% Import Necessary Packages

from keras import layers
from keras import Input, Model
import numpy as np
import pandas as pd

# %% Pull data from SoyNAM Set

""" Data has been pulled from R SoyNAM packages into .csv format.
The data must be imported for pre-processing.
"""
HeightPhen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\HeightPhen.csv",
                        skiprows=1,usecols=1)

HeightGen = np.loadtxt("C:\GRADUATE SCHOOL\Srivastava Group\Soybeans\SoyNAM_Data\HeightGen.csv",
                        skiprows=1,usecols=1)
