# Directional Sparse Filtering: Tensorflow/Keras Implementation

### Implementation Note: 
- GPU support is highly dependent on individual Tensorflow version's support. As of v2.4.1, not all required operations are supported yet so some operation will be delegated to CPU. To use CPU only, set `use_real_proxies=False`. To use a mix of CPU and GPU, set `use_real_proxies=True`.

- Inline decoupling operation is currently not producing the same result as MATLAB presumably due to complex gradient issue on TF. Non-decoupling version is running correctly. From our experience, the performance drop should be minimal. We provided a constraint-projection implementation in lieu of this, if `inline_decoupling` is set to `True`, but this is known to converge to a worse optima than inline decoupling.

## Python Code for the following papers:

K. Watcharasupat, A. H. T. Nguyen, C. -H. Ooi and A. W. H. Khong, "Directional Sparse Filtering Using Weighted Lehmer Mean for Blind Separation of Unbalanced Speech Mixtures," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4485-4489, doi: 10.1109/ICASSP39728.2021.9414336. [[paper]](https://ieeexplore.ieee.org/document/9414336)

A. H. T. Nguyen, V. G. Reju and A. W. H. Khong, "Directional Sparse Filtering for Blind Estimation of Under-Determined Complex-Valued Mixing Matrices," in IEEE Transactions on Signal Processing, vol. 68, pp.  1990-2003, 2020, doi: 10.1109/TSP.2020.2979550. [[paper]](https://ieeexplore.ieee.org/document/9028226)

A. H. T. Nguyen, V. G. Reju, A. W. H. Khong and I. Y. Soon, "Learning complex-valued latent filters with absolute cosine similarity," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 2412-2416, doi: 10.1109/ICASSP.2017.7952589. [[paper]](https://ieeexplore.ieee.org/document/7952589)

## Dependencies
```
- numpy
- tensorflow >= 2.0
- kapre
- lapjv == 1.3.1
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
