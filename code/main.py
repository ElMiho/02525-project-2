from image_helpers import load_mat_file, show_mat_file

import matplotlib.pyplot as plt

for day in ["01", "06", "13", "20", "28"]:
    show_mat_file(
        path_from = f".././files-from-learn/data/multispectral_day{day}.mat",
        path_to = f"./images/day{day}.png"
    )

