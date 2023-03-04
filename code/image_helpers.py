import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import PIL.Image

def load_mat_file(path:str) -> np.ndarray:
    mat = io.loadmat(path)["immulti"]
    mat = np.asarray(mat)
    return mat

def load_image(path:str) -> np.ndarray:
    im = PIL.Image.open(path)
    im = np.asarray(im)
    return im

def show_mat_file(path_from:str, path_to:str) -> None:
    mat = load_mat_file(path_from)
    n_images = mat.shape[2]

    n_rows = int(np.ceil(np.sqrt(n_images)))
    n_cols = int(n_rows)

    plt.figure(figsize=(20,20))
    plt.tight_layout()

    for i in range(n_images):
        im = mat[:,:,i]
        plt.title(f"Channel: {i}", fontsize=12)
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(im)

    plt.savefig(path_to)






# print(load_mat_file(".././files-from-learn/data/multispectral_day01.mat"))


