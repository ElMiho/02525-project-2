#%%
import PIL.Image
import scipy.io as io
from multivariate_functions import *
from tabulate import tabulate

# Import data
folder = ".././files-from-learn/data/"
days = ["01", "06", "13", "20", "28"]
multispectrals = []
annotations = []
for i in days:
    multispectral_day = io.loadmat(folder + f"multispectral_day{i}.mat")["immulti"]
    multispectrals.append(multispectral_day)
    
    annotation_day = PIL.Image.open(folder + f"annotation_day{i}.png")
    annotation_day = np.asarray(annotation_day)
    annotations.append(annotation_day)
#%%
p1 = 0.5
errors = []
cov_inv_list = []
mean_meat_matrix = []
mean_fat_matrix = []

# Train 6 models
for i in range(len(days)):
    model_day = days[i]

    multispectral_current_day = multispectrals[i]
    annotation_current_day = annotations[i]

    multispectral_current_day = np.asarray(multispectral_current_day)
    annotation_current_day = np.asarray(annotation_current_day)

    # Generate model
    cov_inverse, mean_fat_list, mean_meat_list, fat_mask_actual = multivariate_model(annotation_current_day,multispectral_current_day)
    
    test_error = []
    test_error.append(f"Test day: {model_day}")
    for i in range(len(days)):
        test_day = multispectrals[i]
        test_day = np.asarray(test_day)
        fat_mask_predicted = predicted_classification(cov_inverse, mean_fat_list, mean_meat_list, test_day, model_day, p1)
        error = calculate_error(fat_mask_predicted, fat_mask_actual)
        test_error.append(error)

    errors.append(test_error)
#print(errors)
with open('ErrorTable.txt', 'w') as f:
    f.write(tabulate(errors, tablefmt = "latex"))
f.close()