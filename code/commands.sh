# With best hyperparameters

# For Beauty
python train.py model=LightGCN hash_type=full dataset=Beauty
python train.py model=LightGCN hash_type=GraphHash dataset=Beauty resolution=200
python train.py model=LightGCN hash_type=Leiden dataset=Beauty resolution=201
python train.py model=LightGCN hash_type=BACO dataset=Beauty resolution=0.1308

# For Gowalla
python train.py model=LightGCN hash_type=full dataset=Gowalla
python train.py model=LightGCN hash_type=GraphHash dataset=Gowalla resolution=200
python train.py model=LightGCN hash_type=Leiden dataset=Gowalla resolution=197
python train.py model=LightGCN hash_type=BACO dataset=Gowalla resolution=7.57

# For Yelp2018
python train.py model=LightGCN hash_type=full dataset=Yelp2018
python train.py model=LightGCN hash_type=GraphHash dataset=Yelp2018 resolution=200
python train.py model=LightGCN hash_type=Leiden dataset=Yelp2018 resolution=198
python train.py model=LightGCN hash_type=BACO dataset=Yelp2018 resolution=5.501

# For AmazonBook
python train.py model=LightGCN hash_type=full dataset=AmazonBook
python train.py model=LightGCN hash_type=GraphHash dataset=AmazonBook resolution=200
python train.py model=LightGCN hash_type=Leiden dataset=AmazonBook resolution=198
python train.py model=LightGCN hash_type=BACO dataset=AmazonBook resolution=4.733
