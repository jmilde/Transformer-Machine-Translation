# Transformer Translation with SentencePiece 
This is an implementation of the transformer architecture, combined with using SentencePiece for the text processing.
## Getting Started

To get the summarizer to work one first has to install requiered libraries and run some preprocessings

### Prerequisites

Besides Python 3.x and tensorflow 1.3 following libraries are additionally requiered, run the following commands in your shell

```
pip install sentencepiece sacrebleu numpy tqdm
```
The default for all following instructions is to have the project folder as cd.

First run the following commands
```
mkdir ./data ./ckpt ./log ./results
```
In the next step download the german-english corpus from 
```
http://www.statmt.org/europarl/
```
and unpack it into the folder data so that the path is /data/de-eng

## Preprocessing and SentencePiece training
Run the following code to do all requiered preprocessing. Parameters can be adjused in the hyperparameters.py file
```
cd .code/
python3 preprocess.py
```
## Training the Model
to train the transformer simply run the following lines of code in the terminal. Again, parameters can be adjused in the hyperparameters.py file.
```
cd .code/
python3 train.py
```

## Inference
change the "trial" in the paramaters dictionary in hyperparameters.py to the the ckpt you want to load. if you trained a model with trial = transformer, the first checkpoint would be transformer0.
then run in terminal
```
cd .code/
python3 infer.py
```
the results will be automatically saved in the folder results. 

## Evaluation
to evaluate the output of the inference by caluclating the bleu score, just run the following lines in terminal
```
cd ./results
cat tf_norm_pred.txt | sacrebleu testset.txt
```
