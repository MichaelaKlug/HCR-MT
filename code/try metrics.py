import torch
from torch.nn import functional as F
import numpy as np
import queue

# def dice_loss(outputs, targets):
#     num_classes=4
#     smooth = 1e-5
#     loss=0
#     for class_idx in range(num_classes):
#         # Extract the predictions and ground truth for the current class
#         pred_class = outputs[:, class_idx, :, :, :]
#         target_class = (targets == class_idx).float()

#         intersection = torch.sum(pred_class * target_class) #is this correct?

#         union = torch.sum(pred_class) + torch.sum(target_class)
#         # Calculate the Dice coefficient for the current class
#         dice = (2 * intersection + smooth) / (union + smooth)

#         # Add the Dice coefficient for the current class to the loss
#         loss += (1 - dice)

#     # Average the loss over all classes
#     loss /= num_classes

#     return loss

def dice_loss(outputs, targets):
    num_classes = 4
    smooth = 1e-5
    loss = 0

    for class_idx in range(num_classes):
        # Extract the predictions and ground truth for the current class
        #pred_class = outputs[:, class_idx, :, :]
        pred_class=(outputs==class_idx).float()
        print(pred_class)
        target_class = (targets == class_idx).float()
        print(target_class)

        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class)

        dice = (2 * intersection + smooth) / (union + smooth)

        loss += (1 - dice)

    loss /= num_classes

    return loss

outputs=np.array([
  [
    [1, 2, 0],
    [3, 0, 2],
    [1, 3, 2]
  ]
])

targets=np.array([
  [
    [3, 3, 3],
    [3, 0, 2],
    [1, 3, 2]
  ]
])

# outputs=torch.from_numpy(outputs)
# targets=torch.from_numpy(targets)

# print(dice_loss(outputs,targets))
  
# print(np.array([1,2,2])+np.array([1,1,1]))
print([]+[1,2,3]+[4,5,6])
