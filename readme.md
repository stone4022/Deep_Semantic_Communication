# Deep Learning Semantic Communication Systems 

## Requirements
+ See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.


```
## Preprocess
```shell
mkdir data
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar zxvf europarl.tgz
python preprocess_text.py
```

## Train
```shell
python main.py 
```

## Evaluation
```shell
python performance.py
```
### Notes
+ If you want to compute the sentence similarity, please download the bert model.
