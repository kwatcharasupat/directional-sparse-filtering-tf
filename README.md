# Directional Sparse Filtering: Tensorflow/Keras Implementation

## Python Code for the following papers:

K. Watcharasupat, A. H. T. Nguyen, C. -H. Ooi and A. W. H. Khong, "Directional Sparse Filtering Using Weighted Lehmer Mean for Blind Separation of Unbalanced Speech Mixtures," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4485-4489, doi: 10.1109/ICASSP39728.2021.9414336. [[paper]](https://ieeexplore.ieee.org/document/9414336)

A. H. T. Nguyen, V. G. Reju and A. W. H. Khong, "Directional Sparse Filtering for Blind Estimation of Under-Determined Complex-Valued Mixing Matrices," in IEEE Transactions on Signal Processing, vol. 68, pp.  1990-2003, 2020, doi: 10.1109/TSP.2020.2979550. [[paper]](https://ieeexplore.ieee.org/document/9028226)

A. H. T. Nguyen, V. G. Reju, A. W. H. Khong and I. Y. Soon, "Learning complex-valued latent filters with absolute cosine similarity," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 2412-2416, doi: 10.1109/ICASSP.2017.7952589. [[paper]](https://ieeexplore.ieee.org/document/7952589)

## Dependencies
```
- numpy
- tensorflow
- kapre
- lapjv
- fire
- tqdm
```

## To run example script

```
git clone https://github.com/karnwatcharasupat/directional-sparse-filtering-tf.git

cd ./directional-sparse-filtering-tf
```

For ICASSP 2021 (Lehmer Mean):
```
python main.py --input_path path\to\mixture.wav --n_src number_of_sources --mode icassp2021
```

For TSP 2020 / ICASSP 2017 (Power Mean):
```
python main.py --input_path path\to\mixture.wav --n_src number_of_sources --mode tsp2020
```

## Documentation

Coming Soon :)

## For Matlab version:
[https://github.com/e13000/directional_sparse_filtering](https://github.com/e13000/directional_sparse_filtering)
