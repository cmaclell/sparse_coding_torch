import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image

class ScaleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(2064, 128)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class BamcPreprocessor(nn.Module):
    
    # height=600, width=1500, top_crop=0.3, bottom_crop=0.2, y_offset=80, x_scale_span=90 <- Looks good
    def __init__(self, height=400, width=700, top_crop=0.15, bottom_crop=0.05, depth=6, y_offset=80, x_scale_span=90):
        super().__init__()
        self.depth = depth
        if self.depth < 5:
            raise ValueError("depth must be >=5")

        self.top_crop = round(height * top_crop)
        self.bottom_crop = round(height * top_crop)
        self.height = height + self.top_crop + self.bottom_crop
        self.width = width
        self.y_offset = y_offset
        self.x_scale_span = x_scale_span
        self.scale_classifier = ScaleClassifier()
        self.scale_classifier.load_state_dict(torch.load('scale_classifier_weights.pt'))
        self.scale_classifier.eval()

    # x represents our data
    def forward(self, x):
        single = False
        if len(x.shape) == 4:
            single = True
            x = x.unsqueeze(0)
        
        in_height = x.shape[3]
        in_width = x.shape[4]
        
        output = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.height-self.top_crop-self.bottom_crop, self.width)
        
        for i in range(x.shape[0]):
            img = x[i, 0, :, :, :]
            
            slice_frame = img[0, self.y_offset:in_height, :self.x_scale_span]
            slice_frame[slice_frame < 40/254] = 0
            
            y_coord_max = torch.max(torch.where(slice_frame != 0)[0])
            y_coord_min = torch.min(torch.where(slice_frame != 0)[0])
            
            s = slice_frame[y_coord_max-21:y_coord_max, :]
            
            pred = self.scale_classifier(s.unsqueeze(0).unsqueeze(0).float())
            pred_class = torch.argmax(pred, dim=1)[0]
            if pred_class == 0:
                img_size = 12
            elif pred_class == 1:
                img_size = 5
            else:
                img_size = 16
            
            scale_height = y_coord_max - y_coord_min
            
            if img_size == 5:
                extra_px = int(torch.round((scale_height)/5 * (self.depth - 5)).item())
                crop1 = torch.zeros(x.shape[2], scale_height + extra_px, x.shape[4])
                img_size = self.depth
            else:
                crop1 = torch.zeros(x.shape[2], scale_height, x.shape[4])
                
            crop1[:,:scale_height, :] = x[i, 0, :, self.y_offset+y_coord_min:self.y_offset+y_coord_max, :]
            
            resize_height = round(self.height * img_size / self.depth)
            resize_width = round(resize_height / crop1.shape[1] * crop1.shape[2])
            
            resized = T.Resize(size=(resize_height, resize_width))(crop1)
            crop2 = resized[:, :self.height, :]
            
            center_x = round(crop2.shape[2] / 2)
            width_span = round(self.width / 2)
            crop3 = crop2[:, self.top_crop:-self.bottom_crop, center_x-width_span:center_x+width_span]
            
            output[i, 0] = crop3
            
        if single:
            output = output.squeeze(0)
            
        return output
    
    