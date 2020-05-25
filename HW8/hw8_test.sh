cat model/model_* > model.pkl
python3 test.py $1 model.pkl $2
