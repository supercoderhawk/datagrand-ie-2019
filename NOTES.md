# submit notes
## first commit (2019-07-06)

**baseline.txt**

### Configuration:
* split provided training data, 90% for training, 10% for testing
* simple CRF model + ngram features, in `crf_feature.py`

### Result
#### 10% test data
|a  |precision |recall  |   f1 |
|---|----------|--------|------|
|a  |  100.0%  |99.79%  |99.89%|
|b  |  99.66%  |99.87%  |99.76%|
|c  |  100.0%  |99.89%  |99.95%|

#### online
0.859769444039107 

### conclusion
1. training data distribution seems unbiased
2. distribution of test data and training data seems different
3. CRF meets its upper bound
4. next step is NN based method, and MUST use pretrained LM such as ELMo

 