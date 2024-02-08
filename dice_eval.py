import numpy as np
import os
import ants

def binary_dice_coefficient_fixed(y_true, y_pred, label):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_pred_f[y_true_f == label]==label)
    return(-1.0 * (2.0 * intersection)/(np.sum(y_true_f==label) + np.sum(y_pred_f==label)))


testing_volume_folder = "/scratch/gilbreth/li2068/nnUNet_v2/nnUNet_raw/Dataset300_synrad/nnunet_predicted_results"
label_folder = "/scratch/gilbreth/li2068/nnUNet_v2/nnUNet_raw/Dataset300_synrad/labelsTs"

class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
ct = 0 
for each in os.listdir(testing_volume_folder):
    if "case" in each:
        y_pred = ants.image_read(os.path.join(testing_volume_folder,each))
        y_true = ants.image_read(os.path.join(label_folder,each))
        class1_dice = binary_dice_coefficient_fixed(y_true, y_pred, label=1)
        class2_dice = binary_dice_coefficient_fixed(y_true, y_pred, label=2)
        class3_dice = binary_dice_coefficient_fixed(y_true, y_pred, label=3)
        class4_dice = binary_dice_coefficient_fixed(y_true, y_pred, label=4)
        class5_dice = binary_dice_coefficient_fixed(y_true, y_pred, label=5)
        
        class1.append(class1_dice)
        class2.append(class2_dice)
        class3.append(class3_dice)
        class4.append(class4_dice)
        class5.append(class5_dice)
        print(f"processed case {ct}")
        ct+=1

print(f"class 1 average{sum(class1)/len(class1)}")
print(f"class 2 average{sum(class2)/len(class2)}")
print(f"class 3 average{sum(class3)/len(class3)}")
print(f"class 4 average{sum(class4)/len(class4)}")
print(f"class 5 average{sum(class5)/len(class5)}")