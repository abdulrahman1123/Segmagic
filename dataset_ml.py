import hashlib
from pathlib import Path

import numpy as np
import skimage
import tifffile
import torch
import pickle
import matplotlib.pyplot as plt

class TrainImage():
    def __init__(self, info_dict):
        self.info_dict = info_dict

        self.out_channels = [c['name'] for c in self.info_dict['metadata']['channels'] if not c['name'].startswith('p_')]
        if self.info_dict["metadata"]["one_channel"]:
            self.out_channels = [self.out_channels[0]]
        self.labels = self.info_dict["labels"]
    
        self.in_channels = self.info_dict["metadata"]["channels"]

        self.regions = []
        
        
        
        if self.info_dict["metadata"]["use_regions"]:
            region_coords = [f['geometry']['coordinates'][0] for f in self.info_dict['features']if f['properties']['classification']['name'] == 'Region*']
        else:
            region_coords = [[[0, 0],[0, info_dict["metadata"]["width"]],[info_dict["metadata"]["width"], info_dict["metadata"]["height"]], [info_dict["metadata"]["height"],0], [0, 0]]]
        
        
        self.regions.extend(
            {
                "coord": region,
                "mask": None,
                "x": None,
                "y": None,
                "w": None,
                "h": None,
            }
            for region in region_coords
        )

        if "width" in info_dict["metadata"]:
            self.image_width = info_dict["metadata"]["width"]
            self.image_height = info_dict["metadata"]["height"]
        else:
            # use region size
            self.image_width = max(max(p[0] for p in region['coord']) for region in self.regions)
            self.image_height = max(max(p[1] for p in region['coord']) for region in self.regions)

        self.polygons = {c: [f['geometry']['coordinates'] for f in self.info_dict['features'] if f['properties']['classification']['name'] == c] for c in self.labels}
        self.load_mask()

    def generate_hash(self, region):
        # use the hash of image path, region x, y, w, h to generate a filename
        # this way we can cache the region
        
        # get the hash of the image path
        filename_hash = hashlib.sha256(self.info_dict["path"].encode('utf-8')).hexdigest()
        
        # get the hash of the region
        filename_hash += hashlib.sha256(str(region['x']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['y']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['w']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['h']).encode('utf-8')).hexdigest()

        # also include downscale
        filename_hash += hashlib.sha256(str(self.info_dict["metadata"]["downscale"]).encode('utf-8')).hexdigest()
        
        # make hash shorter to use it as filename using sha1
        filename_hash = hashlib.sha1(filename_hash.encode('utf-8')).hexdigest()
        
        return filename_hash

    def cache_region(self, region):
        filename_hash = self.generate_hash(region)

        # save region as pickle to ./cache/{hash}.pkl, overwrite anyway
        cache_path = Path("cache") / f"{filename_hash}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # save the region
        with open(cache_path, 'wb') as f:
            pickle.dump(region, f)

    def load_region_cache(self, info_dict, region):
        return None
        filename_hash = self.generate_hash(region)

        # save region as pickle to ./cache/{hash}.pkl, overwrite anyway
        cache_path = Path("cache") / f"{filename_hash}.pkl"

        # check if cache exists
        if not cache_path.exists():
            return None
        
        # save the region
        with open(cache_path, 'rb') as f:
            region = pickle.load(f)

        return region
    
    def load_region(self, region):
        # try cached region first
        if l_cached := self.load_region_cache(self.info_dict, region) is not None:
            return l_cached

        with tifffile.TiffFile(self.info_dict["path"]) as tif:
            # if labels correspond to each image channel
            if self.info_dict["metadata"]["different_pages"]:
                for i, c in enumerate(self.labels):
                    region["image"][i, :, :] = tif.asarray()[i,:,:][region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
            # if just one image channel or RGB
            else:
                region["image"] = tif.asarray()[region['y']:region['y'] + region['h'], region['x']:region['x'] + region['w']]
                region["image"] = region["image"].transpose(2, 0, 1)
            
            # check if channels are in the correct order
            if region["image"].shape[0] != len(self.in_channels):
                #region["image"] = region["image"].transpose(2, 0, 1)
                pass
                

        # downscale if needed
        if self.info_dict["metadata"]["downscale"] > 1:
            # use bilinear resize for image, and nearest neighbor for mask
            region["image"] = skimage.transform.rescale(
                region["image"],
                (1, 1 / self.info_dict["metadata"]["downscale"], 1 / self.info_dict["metadata"]["downscale"]),
                preserve_range=True,
                anti_aliasing=True
            )

            region["mask"] = skimage.transform.rescale(
                region["mask"],
                (1, 1 / self.info_dict["metadata"]["downscale"], 1 / self.info_dict["metadata"]["downscale"]),
                order=0
            ).astype(np.bool_)

            region["h"] = int(region["h"] / self.info_dict["metadata"]["downscale"])
            region["w"] = int(region["w"] / self.info_dict["metadata"]["downscale"])
            region['x'] = int(region['x'] / self.info_dict["metadata"]["downscale"])
            region['y'] = int(region['y'] / self.info_dict["metadata"]["downscale"])

        # cache the region
        self.cache_region(region)

        return region

    def load_mask(self):
        # for each region, create a mask        
        full_mask = np.zeros((len(self.labels), self.image_height, self.image_width), dtype=np.bool_)
        

        for i, c in enumerate(self.labels):
            for poly in self.polygons[c]:
                if len(poly) == 1:
                    full_mask[i, :, :] += skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in poly[0]])
                else:
                    try:
                        full_mask[i, :, :] += skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in poly[0]])
                    except ValueError as e:
                        # no Polygon here, for example does not work with Multipolygons
                        # maybe implement method here to work with other shapes
                        print(poly)
                        raise ValueError from e
                    for p in poly[1:]:
                        full_mask[i, :, :] ^= skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in p])
        

        for region in self.regions:
            # height, width --> largest x - smallest x, largest y - smallest y
            region['x'] = min(p[0] for p in region['coord'])
            region['y'] = min(p[1] for p in region['coord'])
            region['w'] = max(p[0] for p in region['coord']) - region['x']
            region['h'] = max(p[1] for p in region['coord']) - region['y']

            region["mask"] = full_mask[:, region['y']:region['y']+region['h'], region['x']:region['x'] + region['w']]
            region["image"] = np.zeros((len(self.in_channels), region['h'], region['w']), dtype=np.int16)

            # load the region
            region = self.load_region(region)

    def sample_position(self, width, height):
        # get a random region
        region = np.random.choice(self.regions)
        
        # get a random position within the region
        x = np.random.randint(region['x'], region['x']+region['w']-width)
        y = np.random.randint(region['y'], region['y']+region['h']-height)

        return (x, y, width, height), region

    def load_image(self, region, position):
        x, y, w, h = position
        
        # make sure we don't go out of bounds
        # _w = min(w, self.image_width - x)
        # _h = min(h, self.image_height - y)
        r_x = x - region['x']
        r_y = y - region['y']
        image = region["image"][:, r_y:r_y + h, r_x:r_x + w]
    
        # pad to h, w if needed
        if image.shape[1] < h:
            image = np.pad(image, ((0, 0), (0, h - image.shape[1]), (0, 0)), mode='constant')

        if image.shape[2] < w:
            image = np.pad(image, ((0, 0), (0, 0), (0, w - image.shape[2])), mode='constant')

        return image
    
    def get_mask(self, region, position):
        x, y, w, h = position
        dx = x - region['x']
        dy = y - region['y']
        return region["mask"][:, dy:dy + h, dx:dx + w]


class ImageDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_dict, size=(128, 128), transforms=None, augmentations=None, repeats=16):
        self.data_dict = data_dict
        self.size = size
        # aug and transforms are from albumentations
        self.transform = transforms
        self.augmentation = augmentations
        self.repeats = repeats

    def __len__(self):
        return len(self.data_dict) * self.repeats
        
    def __getitem__(self, index):
        index = index % len(self.data_dict)

        pos, region = self.data_dict[index].sample_position(self.size[0], self.size[1])
        image = self.data_dict[index].load_image(region, pos)
        mask = self.data_dict[index].get_mask(region, pos)

        image = np.float32(image / 2**8)
        # expects channels last
        image = np.transpose(image, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform:
            sample = self.transform(image=image, mask=mask) 
            image, mask = sample['image'], sample['mask']

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return image, np.float32(mask)
