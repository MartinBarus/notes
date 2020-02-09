# Introduction

These are my personal notes of the following ML course 

[*How to Win a Data Science Competition: Learn from Top Kagglers | Advanced Machine Learning Specialization*](https://www.youtube.com/playlist?list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm)


# FEATURE engineering

## Numeric

### Scaling (also Removing Outliers)
 - min-max
 - standardize
 - for KNN (importance), LM (regularization affected too), NNs (gradient) 
 - use only data from range (0.01 percentile, 0.99 percentile) = winsorize (also Removing Outliers)
 - rank (concat train+test) good for NNs too (also Removing Outliers)
 - log-transform np.log(1+x) (also Removing Outliers)
 - raising to power <1, np.sqrt(x+ 2/3) (extract square root, data closer together, values near 0 more distinguishable)

### Feature generation:
 - relative values
 - + - * /
 - fractional part (2.34 -> 0.34)


## Ordinal + Categorical

 - label encoding
 - one hot encoding
 - frequency encoding (possibly followed by rank encoding)


## Date time

### Periodicity
 - second, minute, hour, day, month, week_day, .., is_holiday - repetitive patterns

### Time since particular event
 - row independent - day since 1 Jan 1970 (UNIX timestamp)
 - row dependent - days since last/ left to next holidays/campaign/whatever of specific row

### Difference between dates
 - diff between last purchase date, last call date

## Coordinates 

 - distance to some special areas
 - perhaps split into grid, most expansive flat, for each point distance to the most expansive flat in it's grid
 - mean flat price for area
 - rotate by 45 degrees

 - interesting places
 - centers of clusters
 - aggregated statistics

## Missing Values

look at histogram, spikes of special values could show missing values were replaced by this very common value

### Replace
 - -999/-1/special value bad for LM, NNs
 - mean/median hard for trees to tell
 - isNull flag
 - reconstruct 
 - avoid replacing before feature generation (otherwise stats will be computed using missing)

Encoding unseen test data using encoders (categorical) - use category with closest frequency in train that this has in test


## Text

### Bag Of Words
 - new column for each word
 - value is count/ binary/ tf-idf

 - n-grams - not only words, but n neighbor words as single column
 - lowercase, lemma/stemming, stopwords 


### Embeddings
 - word2vec - each word is a vector (lookup), distance in the space makes sense (man woman -> king queen also Glove/fastText
 - doc2vec
 - pretrained


 # Images

### CNNs
  - fine-tuning
  - augmentation - crops/rotations/noise/color/blur ...



# Exploratory Data Analysis (EDA)
 - Better understand
 - intuition
 - hypothesis
 - insights

1 Getting Domain Knowledge
- understand problem
2 Data is intuitive and agrees with our domain knowledge
- data with reasonable bounds
3 Understand how data was generated 
- (Train test vs Train set difference)
- help set up proper validation

## Exploring anonymized data
- Encode/hash text
- meaningless feature names

Explore individual features
- guess meaning
- guess types

Example - numeric feature with mean ~0 std~0 was most likely standardized. If int type, we can divide by the smallest difference between two consecutive values to get original scale, remove the fractional part (should be mostly the same) (still has offset)



## Visualization

Single Features
- histograms - (peak mean could suggest missing values filled by mean )
- plot (index vs values) `plt.plot(x, '.')`, better `plt.scatter(range(len(x)), x, c=y)`
- statistics: `df.describe()`
- `x.value_counts()`, `x.isnull()`


Feature relations
- `plt.scatter(x1, x2)` // color code by label: class0, class1, test - can show that test is completely different to train data 
- `pd.scatter_matrix(df)`
- `df.corr(), plt.matshow()`
- `df.mean().plot(style='.')` -> `df.mean().sort_values().plot(style='.')`


## Dataset Cleaning:
 - remove constant feature (enough if constant in train)
 - drop duplicated features, for categorical, encode and then compare /pd.Series.factorize(), LabelEncoding based on occurrence order
 - drop duplicated rows 
 - does train/test have common rows?
 - check if dataset is shuffled: plot target vs index

# Validation

## Holdout (sklearn ShuffleSplit)
- OK if lots of data, score similar across holdouts/folds

## K Folds
- good if models differ across folds

## Leave one out
- good if little data, fast enough model

## Stratification
- even label distribution
- good for little data/ unbalanced/ multiclasss


## Data splitting
- set up so that validation reflects train/test set

Random vs TIme based

- differ in generated features/way model relies on features/possible leaks 

### Row based random
- rows are independent 

### Time based
- possibly moving window evaluation 

### ID split
- f.e. pictures from different fishing boats, test boats are different, cluster images first, run validation using clusters 

### Combined


## Validation Problems

### Validation stage
- inconsistent optimal parameters and scores across folds / splits 
- F.e. higher sales around Christmas
- could be too little data (trust your validation)
- could be too diverse and inconsistent data (similar values, different target)
- check your train/test split
- possible solution: 

Extensive Validation
- average results form multiple K folds (different seeds)
- find hyperparams using one Kfold split (seed), evaluate using other 

### Submission stage (submit to kaggle)
- LB is significantly higher/lower than Validation
- LB not correlated with Validation
- first have a look at KFold mean/std of scores, assume it this is expected
- possible reasons: too little data in public leaderboard/train and test from different distribution 


Different Distribution
- calculate mean of target on train distribution, calculate mean of test using LB probing, shift predictions by offset
- ensure validation distribution is same as test distribution (f.e. man/woman)


LeaderBoard shuffle
- Randomness, predictions very similar 
- Little data
- Different Public/Private distribution (often time series)


# Data Leaks
- Test data contains signal, avoid, bad in reality

## TimeSeries 
- using information from future, splits have to be done by time
- User history in CTR tasks
- weather

## Other
- meta information - file origin, etc, f.e. pictures of cats different meta-data - date/quality of dog images
- ID information - maybe a hash, may contain traces of target
- Row Order - data should be shuffled

## Leaderboard probing
- change scores for some subset of rows, to find out their true labels - part of public LB
- by constant scores you can see ratio of private+public LB scores - find 0/1 ratio if test/train splits differ


# Metrics

## Regression

 MSE - optimal is mean
 RMSE vs MSE - same minimizers, not the same gradient -> in gradient methods different optimizing hyperparams,
 R2 - 0 if our model same as best constant MSE (baseline), 1 if perfect, optimizing R2 equivalent to optimizing MSE
 MAE - optimal is median, robust, 2*5$ mistake is same as 10$ (later 4 times worse for rmse), robust to outliers
 MAE - gradient is step function

 MSPE, MAPE - like weighted MSE, MAE, relative to target, f.e. sales on stores, one store sells 1000 items, second 10, err 1 is different for the stores
 MSPE - optimal is weighted mean of target 
 MAPE - optimal is weighted median
 RMSLE - log first + const (zero allowed), then RMSE, asymmetric, also relative error like MSPE,MAPE, always better to predict more

## Classification

Hard prediction - label, soft prediction - probability

- Accuracy - straightforward, hard prediction, best constant: most frequent class -> problem with unbalanced dataset (very high baseline), hard to optimize
- Logloss - posterior probability, in practice clipped between (0+eps, 1-eps), extremely penalizes very wrong answers
- logloss - best consts: probabilities = frequencies of each class
- AUC - tries all/many threshold - probability of
- AUC - TPR vs FPR, easy interp: TP - FP go up for TP, left for FP, area under
- AUC - fraction of correctly ordered pairs (pair has to have 1 of each class), baseline - 0.5, independent of baseline
- Cohen's Kappa: almost: 1-((1-accuracy)/(1-baseline_accuracy)) - 0 if baseline, 1 if perfect accuracy, similar to R2 trick
- Cohen's Kappa: : 1-((1-accuracy)/(1-pE)), pE - what accuracy if we randomly permute predictions, pE = (1/N^2 ) * sum_over_k(nk1 * nk2 )
- Kappa: 1 - (error/baseline_err)
- Weighted Kappa: 1 - (weighted error/weighted baseline_err), uses weights matrix, for linearly ordered  it's simple ((0 1 2), (1 0 1), (2, 1, 0)), can be quadratic etc

## Loss

metric you want to optimize, but it can be hard - optimize accuracy
loss is proxy, that can be directly optimized
after loss is optimized we do hacks/transformations to optimize the metric

## Optimization:

- directly - MSE, logloss etc
- preprocess and optimize different metric - MSPE, MAPE, RMSLE
- optimize other metric and postprocess - Accuracy, kappa 
- custom loss function (f.e. for XGboost)
- early stopping - use metric as early stopping, while model optimizes loss

MSE - L2 loss synonym - optimize directly everywhere
MAE - direct optimization in lightGBM, not in XGboost (non defined 2nd derivative ) - L1, median loss/ quantile loss, Huber loss MSE x MAE loss

MSPE, MAPE - add sample weights 1/yi^2/sum(1/yi^2) MSPE, 1/yi/sum(1/yi) MAPE, use MSE, MAE, sample weights often supported
 - or us df.sample(weights = sample_weights), test set stays as is, re-sample dataset several times, average models' predictions
 - if errors are small, optimize in logarithmic scale

RMSLE - 1. transform target zi = log(yi+1), 2. Fit MSE loss model
 - for test: yi = exp(z_pred_i) - 1 (inverse of above) 



 LOGLOSS - synonym- logistic loss,  optimize directly
  - predictions can be calibrated, if they are not - logistic/isotonic reg/ stacking - 2nd level model will use logloss as metric 

 ACCURACY -  if binary, choose any metric, tune threshold 
  - proxy losses - hinge loss (SVM), logloss

AUC - can be optimized directly using pairwise loss, possible in xgboost/lightgbm, often people use logloss

Quadratic Kappa:
 1.optimize MSE,  2. find right threshold - np.round etc, use validation (like accuracy), or implemented in reading materials

# Mean encoding (Target encoding):
 - target/likelihood encoding
 - compensates inability of GBMs to use high cardinality features
 - possibilities: 
  - mean
  - WoE = ln (Goods/Bads) *100
  - count = sum(target )
  - diff = Goods - Bads

## Regularization

- CV loop: use target mean of other folds (4-5 folds)
- Smoothing: global means vs local mean deals with rarer categories, with CV
- add noise, hard to estimate how much noise, usualy with LV
- Expanding mean - least leakage, no hyperparams, in CatBoost: cumsum/cumcount - cumulative mean up till n-th row/cumulative count, cumsum = cumsum - target

## Extensions
- Regression : mean, median, percentile, distribution bins, std
- multiclass: feature per class 
- many to many: users x apps -> encode for each app, take statistics over all app vectors: min max, mean ...
- time series: mean from previous 2 - days (period) -> rolling statistic
- numeric: bin and treat as categorical 
- combination of features 
- if numeric feature has many split points, we could use splits to bin
- select interactions: if in neighboring nodes, most frequent neighbors could be applied together


# Hyperparameters

## General
- select most influential parameters
- understand how they change training
- tune them (manually + intuition) or automatically (hyperopt)

Hyperparam optimization software - define the hyperparam space

Different efects -> under-fitting/over-fitting/just right

Group of params:
 - Constraint parameters: increasing value imposes more constraints (under fitting) - RED
 - Opposite: the higher the value the more relaxed -> over fitting - GREEN

## Tree based models:
GBM - XGB/LGBM/CAT
RandomForest/ExtraTrees - sklearn
Others -RGF

### XGB/LGBM

max_depth / [mas_depth/num_leaves] GREEN
- if you need larger depth, maybe more feature interactions are needed, look at features, create new ones, start with value 7 (learning time will increase )

subsample/bagging_fraction GREEN 
- between [0, 1] (like dropout)

[colsample_bytree/colsample_bylevel] / feature_fraction GREEN 

[*min_child_weight*/lambda/alpha]/ [min_data_in_leaf/lambda_l1/labmda_l2] RED 
*min_child_weight* one of the most important, optimal [0, 5, 15, 300] - wide range


**eta** - learning rate, **num_rounds** - number of trees GREEN
high learning rate - model will not fit (converge)
small learning rate - model will not converge in time, better generalization
for fixed eta, we can find optimal number of trees - use early stopping
when we have good pair learning rate -  num rounds, we can change this by alpha-> use num_rounds * alpha, eta/alpha - improves

### Random Forest/ Extra Trees
- number_of_trees - important - YELLOW - saturates
- max_depth - GREEN - can be None- unconstrained, start with 7, optimal tends to be higher than for GBMs
- max_featuers - GREEN 
- min_sample_leaf - RED
- criterion - GINI(often better)/ENTROPY

- trees are build independently
- find number of trees first, as they are trained separately, the loss saturates- n>N does not yield better results 
- plot validation err vs number of trees 


### Neural Nets (MLPs)
- number of neurons per layer - GREEN. (64 units default )
- number of layers - GREEN (1-2 layer default)
- try overfitting first
- optimizers: SGD + Momentum RED / [ADAM/ADAGRAD/ADADELTA]  - GREEN/overfitting
- batch size - GREEN (32 or 64)
- learning rate - YELLOW start with big (0.1) decrease to see when it converges, increasing batch size by alpha, you can increase LR by alpha, but add regularization 
- regularization - RED - L1/L2/DropOut/ static dropconnect
- add dropout closer to end of network

static dropconnect - hidden 128, 128, make it 4096, 128, randomly drop 99% of connections between input and first hidden layer (4096)

### Linear Models
C/alpha/lambda - RED - regularization

### Overall
- Don't spend too much time on hyperparams
- be patient, let it run
- average different models

# Practical tips
- store data in npy for numpy, hdf5 for pandas
- pandas stores arrays in 64 bit arrays, downcast to 32 bit
- large datasets can be processed in chunks

- extensive validation (CV) not always needed 
- start with fastest models (LGBM )
- use early stopping 
- don't start with svms/ rfs/ nns - take too long 
- from simple to complex, start wit RF not GBMs (^^ haha, contradiction)
- over fit first, then regularize

## Pipeline
- Understand data (1 day)
- EDA (1-2 days) + define good CV
- feature engineering (until last 4 days)
- modeling (until last 4 days)
- ensembling (last 4 days)

Understanding problem:
- what type, what resources, what software, how big data is
- what hardware is needed
- find software
- create new environment
- what metric

EDA
- plot histograms distributions along train/test
- plot feature values vs time
- plot feature values vs target
- cross tab for categorical and target
- consider univariate predictability metrics AUC,IV,R
- binning numerical features and correlation matrices

Define CV strategy
- create validation approach that best reflects real train/test split 
- consistency is key
- if time is important, CV must respect time
- different entities (customers, products) in train and test
- is it completely random
- combination of the above

Feature engineering
- Image - scaling/shifts/rotations
- sound - Fourier, MFCC, spectrograms, scaling
- text - tf-idf, SVD, stemming, spell check, stopwords, n-grams
- time series - lags, weighted averages, exponential smoothing
- categorical - target encoding, one hot, freq, ordinal, label enc
- numeric - scaling, binning, derivatives, outliers removal, dimensionality reduction
- recommender - transactional history, item popularity, frequency of purchases


Modeling
- Image - CNN
- sound - CNN, LSTM
- text - GBMs, Linear, NB, KNNs, LibFM, LIBFFM
- time series - Autoregressive, ARIMA, linear, GBMs, DL, LSTMs
- categorical - GBMs, Linear, DL, LibFM, LIBFFM
- numeric - GBMs, Linear, DL, SVMs
- recommender - CF, GBMs, DL, LibFM, LIBFFM

Ensembling
- save validation predictions and test predictions
- different ways from averaging to stacking
- small data -simple - avg , good if lower correlation along predictions
- bigger data -> means stacking will work
- stacking repeats feature engineering

## Statistic Features

CTR task
- Example userID, pageID, AdPrice,AdPosition
  - min/max/std price, min/max price position per user-page pair 
  - num pages user visited, most visited page

if no feature to do group by on- use neighbors
- hard to implement
- more flexible
- for prices: number of flats/supermarkets/gyms in neighborhood
- mean distance to neighbors (all, target 1, target 0)

## Matrix Factorization 
Decompose User-item matrix R into User matrix U (with user features), item matrix I (item features), so that U*I = R
Text is similar, dimensionality reduction
Bow, TF IDF, Bow bigrams -> decompose, concatenate vectors 
- lossy transformation
- number of latent factors 5-100
- SVD and PCA
- Truncated SVD (works with sparse Matrices)
- NonNegative Matrix factorization NMF - only to non-negative matrices - ratings, counts etc - better for GBMs
- NMF(X), NMF(log(X+1)) try both

!! Train on joint data, apply separately 


## Feature interactions
- either concatenate
- or apply OHE to both columns, pairwise col multiplication
- real values: multiplication, sum, diff, division
- danger of over-fitting

Example - sums, diffs, multiply, division
  - for each type ^^ try all pairwise combinations, train rf, select most important

Construct features from Decision trees:
 - map each leaf into binary feature (like One Hot that it belongs to a leaf)

sklearn: `tree_model.apply()`
xgboost: `booster.predict(pred_leaf=True)`

## tSNE
- visualization
- depends on hyperaparms - perplexity (5-100)
- transformed coordinates as new features
- stochastic, concatenate train and test
- runs for long time for high number of features
- nontrivial interpretation


# Ensembling

- Averaging

## Bagging
- averaging slightly different versions of the same model (Random Forest)
- can be parallel

Bias errors - under-fitting
variance errors - over-fitting

Bagging fights high variance

Params:
- seed
- row subsample/ bootstrapping
- shuffling
- column sub-sampling
- number of models

## Boosting
- each model is trained taking previous model(s) into account
- prediction is weighted average of multiple models


### Weight based
- Train model, compute errors for each rows, create weights reflecting the error, train new model, using same features, set weights as the new weights, recompute the weights
predictionN = pred0*eta + pred1*eta + ... + predN*eta
- residual error based
-params - eta, num_models, the more models, the smaller eta, model type - has to be able to use weights
- adaboost implements this in sklearn

### Residual based
- each time try to only predict the error of last prediction (residual)
- params:
 - learning rate/eta
 - num estimators
 - row & col subsample
 - input models - trees
 - boosting type - fully gradient/dart 
 - xgboost/lightgbm/h2o GBMs/catboost/sklearn GBM

## Stacking
- predictions of number of models in holdout set, then use different meta model to train on these predictions

How it works:
- have train, valid, test, algos A, B C
- train A, B, C on train, make predictions for valid vA, vB, vC and test tA, tB, tC
- train meta algo M on features [vA, vB, vC] using validation target
- predict test data as M.predict([tA,tB,tC])

Caution:
- respect time base split if time series (train, val, test split)
- diversity of models is important:
 - different model types
 - different features
- performance plateaus after N models
- meta model should be simple

## StackNet
do holdout predictions, stack multiple layers, meta-models on top of metamodels

# Final Tips

## Tips for stacking:
Diversity using algos:
 - use 2-3 different GBMS xgb/lgbm/catboost
 - use 2-3 neural nets: keras/pytorch (3 hiddnen, 2 hidden, 1 hidden layers)
 - use 1-2 extra trees/ Random forest
 - use 1-2 linear models/SVMs
 - use 1-2 KNN models
 - use 1 factorization machine (libfm)
 - if data not big, 1 SVM with nonlinear kernel

Diversity having features:
 - categorical features - different encodings
 - numerical - outliers, binning, derivatives, percentiles
 - interactions, groupby statements, unsupervised techniques

subsequent layers: 
 - make simple
 - pairwise differences 
 - row wise statistics: avg, std
 - standard feature selection technique 
 - for every 7.5 models in 1 layer, we add 1 model in the next layer
 - be mindful of target leakage - pick right k for Kfold CV


## Tips for hyperparameter tunning of GBMs
- tune only depth first, fix learning rate 0.1/ 0.05
- then tune sampling parameters col_sample/ sample
- then tune regularizers
