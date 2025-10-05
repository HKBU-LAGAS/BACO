# BACO
Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems

### Prerequisites

-------------
* Please refer to ```requirements.txt``` to install all the dependencies required for running the algorithm
* Please run ```code/setup.sh``` to compile the core components implemented in Cython.
* Please enter the ```code``` directory before running the command.
### Dataset
* Preprocessed datasets are available at ```code/datasets```

* Raw datasets: $Beauty$ is from the paper $\texttt{DirectAU}$ ([link](https://github.com/THUwangcy/DirectAU)); $Gowalla$, $Yelp2018$, and $AmazonBook$ are from the paper $\texttt{GraphHash}$ ([link](https://github.com/snap-research/GraphHash)).

### Dataset detail

------
We conduct experiments on four datasets:
- **Beauty**: A dataset presenting user feedback and ratings for beauty products.
- **Gowalla**: A social network dataset with user check-in data.
- **Yelp2018**: A dataset containing user reviews and ratings for businesses.
- **AmazonBook**: A dataset containing user reviews and ratings for books.

### Run experiments

------

Example command to run $\texttt{BACO}$:

```python
cd code
python train.py model=LightGCN hash_type=BACO dataset=Gowalla resolution=7.57
```

More commands for running are provided in ```code/commands.sh```, with the best searched hyperparameters.
