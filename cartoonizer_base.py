import torch
import cv2
import numpy as np

def FloatTensor2ndarray(nda):
    nda = 255. * nda.cpu().numpy()
    nda = np.clip(nda, 0, 255).astype(np.uint8)
    return nda

def ndarray2FloatTensor(ftensor):
    ftensor = ftensor.astype(np.float32) / 255.0
    ftensor = torch.from_numpy(ftensor)
    return ftensor
    

class Cartoonizer_base_filter:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "down_sampling": ("INT", {
                    "default": 1, 
                    "min": 0, #Minimum value
                    "max": 2, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }), 
                "num_bilateral": ("INT", {
                    "default": 7, 
                    "min": 1, #Minimum value
                    "max": 10, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }), 
                "lside_bilateral": ("INT", {
                    "default": 5, 
                    "min": 3, #Minimum value
                    "max": 7, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }), 
                "sigmaColor": ("INT", {
                    "default": 20, 
                    "min": 5, #Minimum value
                    "max": 255, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }), 
                "sigmaSpace": ("INT", {
                    "default": 9, 
                    "min": 5, #Minimum value
                    "max": 127, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }), 
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "skyrimpasser"

    def execute(self, image, down_sampling, num_bilateral, lside_bilateral, sigmaColor, sigmaSpace):
                
        image = FloatTensor2ndarray(image)
        res = []
        
        if down_sampling == 0:
            for i in range(image.shape[0]): #Batch
                img = image[i]
                for _ in range(num_bilateral):
                    img = cv2.bilateralFilter(img, 2*lside_bilateral-1, sigmaColor, sigmaSpace)
                img = ndarray2FloatTensor(img)
                img.unsqueeze_(0)
                res.append(img)                
            
        else:
        
            for i in range(image.shape[0]): #Batch
                
                img = image[i]
                
                for _ in range(down_sampling):
                    img = cv2.pyrDown(img)
                for _ in range(num_bilateral):
                    img = cv2.bilateralFilter(img, 2*lside_bilateral-1, sigmaColor, sigmaSpace)
                for _ in range(down_sampling):
                    img = cv2.pyrUp(img)
                
                img = ndarray2FloatTensor(img)
                img.unsqueeze_(0)
                res.append(img) 
        
        res = torch.cat(res) 
        return (res,)
        
NODE_CLASS_MAPPINGS = {
    "Cartoonizer base filter": Cartoonizer_base_filter,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cartoonizer base filter": "🎨 Cartoonizer base filter",

}
