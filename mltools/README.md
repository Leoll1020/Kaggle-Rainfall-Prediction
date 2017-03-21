# Python Machine Learning Library

This library provides some simple machine learning algorithms intended mainly for
instructional purposes, written by 
Prof. [Alexander Ihler](http://www.ics.uci.edu/~ihler/)
in collaboration with Austin Sherron.

## Contents

Coverage is mainly of supervised learning algorithms, and organized by learner type.

### Supervised learning
- K-Nearest Neighbor Methods (`knnClassify`, `knnRegress`)
- Bayes Classifiers (`gaussClassify`)
- Linear Regression (`linearRegress`)
- Linear Classification (`linearClassify`)
- Neural Networks (`nnetClassify`, `nnetRegress`)
- Decision Tree Models (`treeClassify`, `treeRegress`)

- Logistic Mean Squared Error Classifier (`LogisticMSEClassify`) and logistic regression?

#### Ensembles of learners
- Bagged Classifier (`BaggedClassify`)
- Gradient Boosting (`GradBoost`)
- Adaptive Boosting (`AdaBoost`)

### Unsupervised Learners (Clustering)
- Hierarchical Agglomerative Clustering (`agglom_cluster`)
- Expectation-Maximization Clustering (`em_cluster`)
- K-Means Clustering (`kmeans`)


### Utilities

#### Data Input/Generation
- Load from CSV text file (`loadtxt`)
- Simple two-component Gaussian model (`data_gauss`)
- Gaussian mixture model (`data_GMM`)
- GUI data entry function (`data_mouse`)

#### Data & Learner Visualization
- Pair plot of data features (`plotPairs`)
- Histogram plot with multiclass data (`histy`)
- Visualize classification & decision boundary (`plotClassify2D`)
- Plot Gaussian distribution center & shape (`plotGauss2D`)

#### Data Utilities
- Randomly shuffle data order (`shuffleData`)
- Split data into training / validation / test (`splitData`)
- Extract cross-validation fold from dataset (`crossValidate`)
- Bootstrap sample from dataset (`bootstrapData`)
- 1-Hot or 1-of-K transformation of discrete data (`to1ofK`, `from1ofK`)
- Convert arbitrary discrete data values to zero-based index (`toIndex`, `fromIndex`)

#### Feature Transformations
- Rescale data (zero center, unit variance) (`rescale`)
- Whiten (center, rotate and rescale) data (`whiten`)
- Feature reduction by random hashing (`fhash`)
- Kitchen sink feature generator (`fkitchensink`)
- Linear Discriminant Analysis features (`flda`)
- Polynomial feature expansion (`fpoly`, `fpoly_mono`)
- Random linear subspace projections (`fproject`)
- Random subset feature selection (`fsubset`)
- Eigenvalue / SVD feature selection (`fsvd`)
- Imputation of missing data entries (`imputeMissing`)

## Todos
- Neural network code
- Ensembles
- Miscellanous later items

### General

## Potential Bugs

- `Y` may be flat (M,), vector (M,1), or multidimensional (M,C); make consistent & add checking & conversion

