# griPEER

MATLAB Toolbox for Fitting Generalized Linear Models with Penalties (called: "penalized") is required to use function griPEER. This toolbox, together with documentation, could be found at https://www.jstatsoft.org/article/view/v072i06 . It is enought to download the files, move to the directory containing toolbox from Matlab level and type: install_penalized .

# Files is this directory
**griPEER.m**      - The implementation of our methodology

**WorkingExample.m** - This code generates the data and presents how to use our method. We recommend to open this script for a quick start.

**griPEER_results.m** - Starts from the data provided in **data** directory to generate the real data analysis which we reported in the preprint. The path to **data** directory needs to be provided at the beginning of the srcipt, the outputs are also saved there.

**generatePlot.R** - R code for generating the plot **HIV_males_results.pdf** presented in https://www.jstatsoft.org/article/view/v072i06. The path to **data** directory needs to be provided at the beginning of the srcipt
