python3 saliency.py $1 model3.pkl $2
python3 filter.py $1 model3.pkl $2
python3 cm.py $1 model1.pkl $2
python3 lime.py $1 model1.pkl $2
python3 shap.py $1 model3.pkl $2 > /dev/null
