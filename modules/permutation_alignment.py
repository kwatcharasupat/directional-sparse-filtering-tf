import numpy as np
import lapjv
from queue import Queue
from multiprocessing import Pool
import multiprocessing as mproc


def permAlign_fineTune_workers(
    f, P, P_normalized, old_P_normalized, total_per
):  # For multithreading through permalign finetuning processes
    [n_freq, n_frame, n_src] = P.shape
    # print("Processing freq bin "+str(f)+" out of "+str(n_freq))

    ## Permutation alignment 7 ##

    # Ignore dc
    f_harmonic_ = np.around(
        np.array(
            [
                (f - 1) / 2,
                2 * (f - 1),
                (f - 1) / 4,
                4 * (f - 1),
                (f - 1) / 8,
                8 * (f - 1),
                (f - 1) / 16,
                16 * (f - 1),
                (f - 1) / 32,
                32 * (f - 1),
            ]
        )
    ).astype(int)
    f_harmonic = np.append(f_harmonic_, [f_harmonic_ + 1, f_harmonic_ + 2])
    f_adjacent = np.append(f - 3, [f - 2, f - 1, f + 1, f + 2, f + 3])
    f_finetuning = np.unique(np.append(f_harmonic, f_adjacent))
    illegal_freq = np.logical_or(
        f_finetuning < 2, f_finetuning > n_freq, f_finetuning == f
    )

    # Integer arrays cannot process Nan: Removing illegal freqs via type conversion
    f_finetuning_float = f_finetuning.astype(float)
    f_finetuning_float[illegal_freq] = np.nan
    f_finetuning_float = f_finetuning_float[~np.isnan(f_finetuning_float)]
    f_finetuning = f_finetuning_float.astype(int)

    # Convert [n_freq...2] to [n_freq-1...1] --> Python indices syntax
    f_py = f - 1
    f_finetuning_py = f_finetuning - 1

    ## Find average Q matrix amongst adjacent and harmonic frequencies ##
    pf = np.reshape(old_P_normalized[f_py, :, :], [n_frame, n_src])
    pf_finetuning = np.reshape(
        np.mean(old_P_normalized[f_finetuning_py, :, :], 0), [n_frame, n_src]
    )
    pf_finetuning = np.divide(
        pf_finetuning, np.std(pf_finetuning, axis=0) + np.finfo(float).eps
    )

    Q = 1 - np.asmatrix(pf).getH() @ np.asmatrix(pf_finetuning) / n_frame
    per, _, _ = lapjv.lapjv(Q)
    # print("Processed freq bin "+str(f)+" out of "+str(n_freq))
    return [f_py, per]


def permAlign_workers(
    i, step, n_freq, P, P_normalized, total_per
):  # For multithreading through permalign processes
    [n_freq, n_frame, n_src] = P.shape
    # print("Processing...")

    ## Permutation alignment 7 ##

    group1_idx = np.arange(i, i + step / 2).astype(int)
    if ((i + step) == (n_freq - 1)) and (step > 2):
        group2_idx = np.arange(i + step / 2, n_freq).astype(int)
    else:
        group2_idx = np.arange(i + step / 2, i + step).astype(int)

    f_1group = np.reshape(np.mean(P_normalized[group1_idx, :, :], 0), [n_frame, n_src])
    f_2group = np.reshape(np.mean(P_normalized[group2_idx, :, :], 0), [n_frame, n_src])

    f_1group = np.divide(f_1group, np.std(f_1group, axis=0) + np.finfo(float).eps)
    f_2group = np.divide(f_2group, np.std(f_2group, axis=0) + np.finfo(float).eps)

    Q = 1 - np.asmatrix(f_1group).getH() @ np.asmatrix(f_2group) / n_frame

    per, _, _ = lapjv.lapjv(Q)

    # Matlab equivalent functions:
    # 		P(group1_idx, :, per) = P(group1_idx, :, :);
    # 		P_normalized(group1_idx, :, per) = P_normalized(group1_idx, :, :);
    # 		total_per(group1_idx, per) = total_per(group1_idx, :);
    # print("Processed freq bins "+str(group1_idx)+" out of "+str(n_freq))

    return [group1_idx, per]


def permutation_alignment7(P_input, max_iter, tol, proc_limit):

    P = np.copy(P_input)
    [n_freq, n_frame, n_src] = P.shape
    total_per = np.tile(np.arange(n_src), [n_freq, 1])

    ## Normalize all sequence to zero-mean unit-variance along time index ##
    P_normalized = P.astype(float)
    P_temp = np.mean(P_normalized, 1)
    P_temp = np.expand_dims(P_temp, axis=1)
    P_normalized = np.subtract(P_normalized, P_temp)
    step = 2
    while step < n_freq:
        permAlign_args = []
        for i in range(1, n_freq, step):  # range(n_freq, 1, -1) gives [1...n_freq-1]
            permAlign_args.append([i, step, n_freq, P, P_normalized, total_per])

        with Pool(processes=proc_limit) as pool:

            pool_outputs = pool.starmap(permAlign_workers, permAlign_args)

        P_temp = np.copy(P)
        P_normalized_temp = np.copy(P_normalized)
        total_per_temp = np.copy(total_per)

        for i in range(len(pool_outputs)):
            group1_idx = pool_outputs[i][0]
            # print(group1_idx)
            per = pool_outputs[i][1]
            for j in range(n_src):
                P[group1_idx, :, per[j]] = P_temp[group1_idx, :, j]
            for j in range(n_src):
                P_normalized[group1_idx, :, per[j]] = P_normalized_temp[
                    group1_idx, :, j
                ]
            for j in range(n_src):
                total_per[group1_idx, per[j]] = total_per_temp[group1_idx, j]

        step = step * 2

    ## Fine tuning w.r.t adjacent and harmonic frequencies ##
    iter = 0
    while True:
        iter = iter + 1
        old_P_normalized = P_normalized

        finetune_args = []

        for f in range(n_freq, 1, -1):
            finetune_args.append([f, P, P_normalized, old_P_normalized, total_per])

        with Pool(processes=proc_limit) as pool:
            pool_outputs = pool.starmap(permAlign_fineTune_workers, finetune_args)

        P_temp = np.copy(P)
        old_P_normalized_temp = np.copy(old_P_normalized)
        total_per_temp = np.copy(total_per)

        for i in range(len(pool_outputs)):
            f_py = pool_outputs[i][0]
            per = pool_outputs[i][1]
            for j in range(n_src):
                P[f_py, :, per[j]] = P_temp[f_py, :, j]
            for j in range(n_src):
                P_normalized[f_py, :, per[j]] = old_P_normalized_temp[f_py, :, j]
            for j in range(n_src):
                total_per[f_py, per[j]] = total_per_temp[f_py, j]

        ## Stop when P wont change anymore ##
        relative_error = np.linalg.norm(old_P_normalized[:] - P_normalized[:]) / (
            np.linalg.norm(old_P_normalized[:] + np.finfo(float).eps)
        )
        stop_flag = (iter > max_iter) or (iter > 1 and relative_error < tol)
        
        if iter > max_iter:
            print("max iter reached")
            
        if (iter > 1 and relative_error < tol):
            print("tolerance reached")
        
        if stop_flag:
            break

    pf_ac = np.reshape(np.mean(P_normalized[1:, :, :], 0), [n_frame, n_src])
    pc_ac = np.divide(pf_ac, np.std(pf_ac, axis=0) + np.finfo(float).eps)
    pf_dc = np.reshape(P_normalized[0, :, :], [n_frame, n_src])

    Q = 1 - np.asmatrix(pf_dc).getH() @ np.asmatrix(pf_ac) / n_frame
    per, _, _ = lapjv.lapjv(Q)  # Solving linear assigment problem => permutation

    P_temp = np.copy(P)
    P_normalized_temp = np.copy(P_normalized)
    total_per_temp = np.copy(total_per)
    for j in range(n_src):
        P[group1_idx, :, per[j]] = P_temp[group1_idx, :, j]
    for j in range(n_src):
        P_normalized[group1_idx, :, per[j]] = P_normalized_temp[group1_idx, :, j]
    for j in range(n_src):
        total_per[group1_idx, per[j]] = total_per_temp[group1_idx, j]
    return P, total_per


if __name__ == "__main__":
    pass
