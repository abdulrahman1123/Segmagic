from dataset_ml import TrainImage, ImageDataLoader
import json
from tqdm.auto import tqdm
import os
from sklearn.model_selection import train_test_split
import pickle
import albumentations as A
import torch
import matplotlib.pyplot as plt



class DataLoader():
    def __init__(self, base_path, labels, project_data_dict):
        self.training_data = []
        self.inference_data = []
        self.base_path = base_path
        self.labels = labels
        self.positions = os.listdir(self.base_path + "/QuPath/data")
        
        

        print('Loading data:')

        for pos in tqdm(self.positions): 
            data = {}                                         
            # get corresponding paths
            data_dir = f"{self.base_path}/QuPath/data/{pos}/server.json"
            
            # get meta information, e.g. image size
            with open(data_dir) as f:
                data_server = json.load(f)
            # merge
            data["metadata"] = {**project_data_dict, **data_server["metadata"]}
            data["labels"] = self.labels
            file_id = data["metadata"]["name"]
            data["file_id"] = file_id
            data["image_name"] = file_id
            
            #open corresponding geojson file
            _file = f"{self.base_path}/Annotations/{file_id}.geojson"
            with open(_file) as f:
                data_geo = json.load(f)
            data["features"] = data_geo["features"]
            
            image_path = f"{self.base_path}\\Images\\{data['file_id']}"
                
            if len(data["features"]) > 0:
                data["path"] = image_path
                
                d = TrainImage(data)
                self.training_data.append(d)
                self.inference_data.append(d)
            else:
                self.data["path"] = image_path
                d = TrainImage(data)
                self.inference_data.append(d)
              

        with open("train_data.pkl", "wb") as td:
            pickle.dump(self.training_data, td)
            
        with open("inference_data.pkl", "wb") as td:
            pickle.dump(self.inference_data, td)

        self.train_valid_split()
        
    
    def train_valid_split(self, ratio=0.2):
        self.train_data, self.valid_data = train_test_split(self.training_data, test_size=ratio, random_state=42)
        print('split into training: '+ str(len(self.train_data)) + ', and validation: ' + str(len(self.valid_data)))
    
    def train_test_split(self, test_ratio=0.1, valid_ratio=0.2, random=42):    
        self.train_val_data, self.test_data = train_test_split(self.training_data, test_size=test_ratio, random_state=42)
        self.train_data, self.valid_data = train_test_split(self.train_val_data, test_size=valid_ratio, random_state=random)
        print('split into training: '+ str(len(self.train_data)) + ', valid: ' + str(len(self.valid_data)) + ', and testing: ' + str(len(self.test_data)))
        print(self.valid_data[-1].info_dict["image_name"])
    
    def run_data_loader(self, kernel_size=256, batch_size=4):
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            #A.GaussNoise(var_limit=(0.005, 0.01), p=0.5),
            #A.RandomBrightnessContrast(p=0.25),
            #A.RandomGamma(p=0.25),
            A.PixelDropout(p=1, per_channel=True),
        ])

        self.train_ds = ImageDataLoader(self.train_data, (kernel_size, kernel_size), augmentations=augmentations)
        self.valid_ds = ImageDataLoader(self.valid_data, (kernel_size, kernel_size), augmentations=None)

        self.train_dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
        self.valid_dl = torch.utils.data.DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    
    def show_example_batch(self):
        batch = next(iter(self.train_dl))
        image = batch[0].numpy()
        mask = batch[1].numpy()

        plt.imshow(image[0,2,...])
        plt.imshow(mask[0,0,:,:], alpha=0.4)
        plt.show()