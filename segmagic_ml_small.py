import torch
import math
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

        

class Segmagic():
    def __init__(self, model_folder='', fast_mode = False):
        self.model_folder = model_folder + '/model'
        self.fast_mode = fast_mode
        try:
            os.mkdir(self.model_folder)
        except:
            pass
        self.model_path = self.model_folder +'/best_model.pth'
        self.ensemble = False
    
        
    def load_model(self):
        print(self.model_folder)
        filepaths = glob.glob(f"{self.model_folder}/*.pth")
        if len(filepaths) > 0 and not self.fast_mode:
            self.ensemble = True
            self.models = []
            for filepath in filepaths:
                if torch.cuda.is_available():
                    model = torch.load(filepath)
                    model.eval()
                    model.cuda()
                else:
                    model = torch.load(filepath, map_location=torch.device('cpu'))
                    model.eval()
                
                self.models.append(model)

            #self.fmodel, self.params, self.buffers = combine_state_for_ensemble(self.models)
        else:
            if torch.cuda.is_available():
                self.model = torch.load(filepaths[0])
                self.model.eval()
                self.model.cuda()
            else:
                self.model = torch.load(filepaths[0], map_location=torch.device('cpu'))
                self.model.eval()


    
    # from http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/
    def dnorm(self, x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def gaussian_kernel(self, size, sigma=1, verbose=False):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    
        kernel_2D *= 1.0 / kernel_2D.max()
    
        if verbose:
            plt.imshow(kernel_2D, interpolation='none', cmap='gray')
            plt.title("Kernel ( {}X{} )".format(size, size))
            plt.show()
    
        return kernel_2D

    def weight_function(self):
        return self.gaussian_kernel(self.kernel_size, sigma=self.kernel_size/5)

    def predict_image(self, image_to_predict, labels, threshold=0.6, kernel_size=256, show=False):
        self.kernel_size = kernel_size
        STEP_SCALE = 0.5
        STEP_SIZE = int(self.kernel_size * STEP_SCALE)
        #image_to_predict = image_to_predict.transpose(2, 0, 1)
        image_width = image_to_predict.shape[2]
        image_height = image_to_predict.shape[1]

        self.load_model()
        x_padding = STEP_SIZE - image_width % STEP_SIZE
        y_padding = STEP_SIZE - image_height % STEP_SIZE


        # fill predicted mask to a tile size
        predicted_mask = np.zeros((
            image_height + y_padding,
            image_width + x_padding,
            len(labels)
        ))

        predicted_weighting = np.zeros((
            image_height + y_padding,
            image_width + x_padding
        ))

        weight = self.weight_function()
        x_steps = image_width // STEP_SIZE
        y_steps = image_height // STEP_SIZE

        # reading image, adjust code here for new images
        #img = image_to_predict.load_image(image_to_predict.regions[0], (0,0,image_to_predict.image_height,image_to_predict.image_width))
        img = image_to_predict.copy()
        img = np.float32(img / 2**8)

        #padding image
        img = np.pad(img, ((0,0), (int(y_padding/2), math.ceil(y_padding/2)), (int(x_padding/2), math.ceil(x_padding/2))), mode="edge")

        pbar = tqdm(total=x_steps * y_steps)
        with torch.no_grad():
            for x in range(x_steps):
                for y in range(y_steps):
                    x0 = x * STEP_SIZE
                    y0 = y * STEP_SIZE
                    x1 = self.kernel_size
                    y1 = self.kernel_size
                    
                    img_tile = img[:,y0:y0+y1, x0:x0+x1]
                    
                
                    img_tile = torch.from_numpy(img_tile).unsqueeze(0)
                    if self.ensemble:
                        outputs = []
                        for model in self.models:
                            if torch.cuda.is_available():
                                pred_m = model(img_tile.cuda())
                            else:
                                pred_m = model(img_tile)
                            outputs.append(pred_m)
                            
                        pred = sum(outputs) / len(outputs)
                        #_, pred = torch.max(ensemble_output, 1)
                        
                    else:
                        if torch.cuda.is_available():
                            pred = self.model(img_tile.cuda())
                        else:
                            pred = self.model(img_tile)

                    pred = pred.squeeze(0).sigmoid().cpu().numpy()
                    
                    pred = pred.transpose(1, 2, 0)
                    
                    predicted_mask[y0:y0+y1, x0:x0+x1,:] += pred*np.expand_dims(weight, axis=2)
                    predicted_weighting[y0:y0+y1, x0:x0+x1] += weight
                    pbar.update(1)

        pbar.close()

        predicted_mask /= predicted_weighting[..., None]

        # remove padding
        predicted_mask = predicted_mask[int(y_padding/2):image_height+math.ceil(y_padding/2), int(x_padding/2):image_width+math.ceil(x_padding/2)]

        # uncertainty measure
        # numpy array p_hat with dimension (N_uncertain, width, height, depth)
        
        p_hat = np.expand_dims(predicted_mask, axis=0)
        epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
        aleatoric = np.mean(p_hat*(1-p_hat), axis=0)
        # Add uncertainties
        uncertainty = epistemic + aleatoric
        # Scale to 1 max overall
        uncertainty /= 0.25

        predicted_mask = np.uint8((predicted_mask>threshold)*255)

        if show:
            for label in range(len(labels)):
                # make subfigures and also show the uncertainty

                fig, axs = plt.subplots(1,2, figsize=(10,5))
                axs[0].set_title('Prediction')
                axs[0].imshow(image_to_predict[0,:,:], cmap='gray')
                axs[0].imshow(predicted_mask[..., label], alpha=0.4, cmap="inferno")

                axs[1].set_title('Uncertainty')
                axs[1].imshow(uncertainty[..., label], cmap="gray")
                plt.show()

        return predicted_mask, uncertainty
    

    
