import numpy as np
from sklearn import metrics
from single_spectral_band import salami_mask

def S(x, mean, cov_matrix_inv, p):
    #Calculate threshold as in equation 23
    res = np.dot(x, cov_matrix_inv @ mean) - 1/2 * np.dot(mean, cov_matrix_inv @ mean) + np.log(p)
    return res

def multivariate_model(annotation_current_day, multispectral_current_day):
    #Returns the model of the given day: [cov_inv, mean_fat_list, mean_meat_list]
    data_vecs = []
    mean_fat_list = []
    mean_meat_list = []

    unknown_channel_mask = (annotation_current_day[:,:,0] > 0)
    fat_channel_mask = (annotation_current_day[:,:,1] > 0)
    meat_channel_mask = (annotation_current_day[:,:,2] > 0)
    
    fat_mask_actual = np.logical_and(fat_channel_mask, salami_mask)
    meat_mask_actual = np.logical_and(meat_channel_mask, salami_mask)

    for i in range(19):
        channel = multispectral_current_day[:,:,i]

        masked_channel_meat = channel[meat_mask_actual]
        masked_channel_fat = channel[fat_mask_actual]
        salami_channel = channel[salami_mask]
        data_vecs.append(salami_channel.flatten())

        mean_meat = masked_channel_meat.mean()
        mean_fat = masked_channel_fat.mean()

        mean_fat_list.append(mean_fat)
        mean_meat_list.append(mean_meat)

    mean_fat_list = np.asarray(mean_fat_list)
    mean_meat_list = np.asarray(mean_meat_list)

    data_vecs = np.asarray(data_vecs)
    cov_matrix = np.cov(data_vecs)
    cov_inverse = np.linalg.inv(cov_matrix)
    
    return cov_inverse, mean_fat_list,mean_meat_list, fat_mask_actual


def predicted_classification(cov_inverse, mean_fat_list, mean_meat_list, multispectral_current_day, model_day, p1):
    #The given model is: [cov_inverse,mean_fat_list,mean_meat_list]
    #Returns the predicted fat level of given model day. 
    p2 = 1-p1
    fat_mask_predicted = np.zeros((514,514))
    for i in range(514):
        for j in range(514):
            if not salami_mask[i,j]: continue
            x = multispectral_current_day[i,j,:]
            # Calculate S1 and S2
            s1 = S(x,mean_fat_list, cov_inverse, p1)
            s2 = S(x,mean_meat_list, cov_inverse, p2)
            if s1 >= s2:
                fat_mask_predicted[i,j] = 1
    return fat_mask_predicted

def calculate_error(fat_mask_predicted, fat_mask_actual):
    # Returns the error
    confusion_matrix = metrics.confusion_matrix(fat_mask_actual.astype(np.int32).flatten(), fat_mask_predicted.astype(np.int32).flatten())
    error = 1 - (confusion_matrix[0,0] + confusion_matrix[1,1])/confusion_matrix.sum()
    return error