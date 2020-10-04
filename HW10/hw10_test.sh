if [[ $2 == *"best"* ]];then
	python3 test_vae_kmeans.py $1 cvae $2 $3
else
	python3 test_vae.py $1 cvae $2 $3
fi
