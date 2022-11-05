import numpy as np
from PIL import Image
from skimage.transform import resize
import os

class ImageDataset:
    def __init__(self, path, target_size):
        self.path = path
        self.target_size = target_size
        self.classes = list(enumerate(os.listdir(path)))
        self.x = []
        self.y = []
        for i, folder in self.classes:
            for image in os.listdir(os.path.join(path, folder)):
                array_of_pixels = self._get_array_of_pixels(os.path.join(path, folder, image))
                self.x.append(array_of_pixels)
                self.y.append(i)
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def _get_array_of_pixels(self,filename):
        """Return an numpy array of pixels from an image file."""
        img = Image.open(filename)
        img = img.resize(self.target_size)
        arr = np.array(img,dtype=np.float32)
        return np.transpose(arr,(2,1,0))

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)