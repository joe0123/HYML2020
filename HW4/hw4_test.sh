wget https://www.dropbox.com/s/rnxjmrpv64atxbf/model4.pkl?dl=1
mv model4.pkl?dl=1 model.pkl
python3 test.py $1 model.pkl $2
