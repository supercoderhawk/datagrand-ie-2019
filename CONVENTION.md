# Conventions in NER Model

## general convention
### project structure convention
#### 1. all experiment related data files store in `data/` folder

under `data/` folder
1. `model/` folder store model files
2. `evaluation/` folder store evaluation related files (prediction result in many scenarios)
3. `datagrand-ie-2019` folder store modelhub data to upload to modelhub server
4. training, validation and testing files store in `data/` folder directory directly
5. other folders and files can create by demand

#### 2. code structure convention
1. `scripts/` folder 
Please put all temporary, informal data process and data runner scripts in this folder!! 
2. `notebooks/` folder
Please put all jupyter notebooks in this folder.

## Named Entity Recognition Related Project Convention
### 1. internal entity data schema

The data store and exchange in project can be unified as follow:
```json
[
  {
    "id": "1",
    "entities": [
                  {"id": "1", "entity":"Google","start": 0,"end": 6, "type": "COMPANY"}
                ],
    "text": "Google is a great company.",
    "tokens": [
    {"text": "Google", "start": 0, "end": 6,"pos_tag": "PROPN"},
    {"text": "is", "start": 7, "end": 9, "pos_tag": "VERB"}, 
    {"text": "a", "start": 10, "end": 11,"pos_tag": "DET"},
    {"text": "great", "start": 12, "end": 13,"pos_tag": "ADJ"}, 
    {"text": "company", "start": 14, "end": 21,"pos_tag": "NOUN"},
    {"text": ".", "start": 22, "end": 23,"pos_tag": "PUNCT"}
     ],
    "labels": ["U-COMPANY", "O", "O", "O", "O", "O"]      
  }
]
```

Note:
1. Every item in above list can be considered as a sentence or paragraph object.
2. **`entities` and `text` are required, other fields are optional.**  
    1. `id` isn't used in most scenarios
    2. `tokens` contain token object, which is a dict including text, start, end and pos_tag. 
    4, `labels` is predicted or true labels of corresponding tokens, used in annotated data to train, validate or test
3. This format is used in evaluation, highlight and other related task.

### 2. model name and prediction data name
#### Model Name Convention

Model file path = `MODEL_DIR + model_name + '.jl'`.

The `model_name` indicates the feature sets name and suffix, which indicates the training data information at mostly.

#### Prediction Data Name
ALL prediction result places in `EVALUATION_DIR`.

File name `model_name + '_' + ext_name + '.json'`

`ext_name` indicates the data source of result, should be in `validation`, `validation_oov`, `test` and `test_oov`
  
### 3. data annotation pipeline

For some project, our data need to manually annotate raw patents by brat. 

The brat annotation pipeline is as follow:  
 
    raw patent (with autolabeled entities)
        => whether need normalize text ? (remove whitespaces, normalize some special characters)
            => split patent into paragraphs by line break
                => aggregate some paragraphs into a txt file accroding to text size (brat will slow down when text is 
                large) 
                    => transform into brat format

**All progresses in above pipeline have implemented**                      
### 4. data preparation pipeline
For most project, our data used for experiment is from manually annotated text from brat. 

Therefore, the start of our pipeline assume brat data.

data process pipeline is as follow:


    brat
        => json (sentence based) 
            => shuffle and split into training, validation and test 
                => json2label (output conll)  


## Code Convention
1. **all file folder value must be end with `/`!!!**

## Other Convention
1. Label Convention
    
Labels must be in such format: **tag + '-' + entity type name**.

Tag can be `BIO` and `BILOU`, entity type must be in title format (First character of token is uppercase, others are 
lowercase)     
**Default Tag is `BILOU`**
 
Example:
* `B-Company`
* `I-DiseaseClass`
* `O`
 