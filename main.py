import tensorflow as tf
import tensorflow.keras as tfk
import fire
import soundfile as sf
from modules.dsf import LehmerMeanDSF

def do_bss(input_path, n_src):
    
    tf.debugging.enable_check_numerics()
    
    x, fs = sf.read(input_path, always_2d=True)
    
    assert x.shape[1] > 1, "Multichannel signal required"
    
    dsf = LehmerMeanDSF(x, n_src=n_src, fs=fs)
    dsf.fit()


if __name__ == "__main__":
    fire.Fire(do_bss)