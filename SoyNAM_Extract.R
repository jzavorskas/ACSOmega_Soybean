## Start Date: 10/25/2022
## Last Edit: 10/25/2022

# The purpose of this file is to extract data from
# the SoyNAM package for use in Python.

library(SoyNAM)

yieldHold = BLUP(trait="yield",family="all",env="all",
                 dereg=FALSE,MAF=0.05,use.check=TRUE,
                 impute="FM",rm.rep=TRUE)

heightHold = BLUP(trait="height",family="all",env="all",
                 dereg=FALSE,MAF=0.05,use.check=TRUE,
                 impute="FM",rm.rep=TRUE)

proteinHold = BLUP(trait="protein",family="all",env="all",
                 dereg=FALSE,MAF=0.05,use.check=TRUE,
                 impute="FM",rm.rep=TRUE)

oilHold = BLUP(trait="oil",family="all",env="all",
                 dereg=FALSE,MAF=0.05,use.check=TRUE,
                 impute="FM",rm.rep=TRUE)

sizeHold = BLUP(trait="size",family="all",env="all",
                 dereg=FALSE,MAF=0.05,use.check=TRUE,
                 impute="FM",rm.rep=TRUE)

SoyData = c("yield"=yieldHold,"height"=heightHold,
            "protein"=proteinHold,"oil"=oilHold,
            "size"=sizeHold)

