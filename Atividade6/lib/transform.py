import numpy as np
import pandas as pd

class Z_score:
    x_mean = None
    x_std = None
    data  = None

    def fit(self, x):
        self.data = pd.DataFrame(x)
        self.x_std = np.asarray(self.data.std())
        self.x_mean = np.asarray(self.data.mean())

    def transform(self, x):
        x_np = np.asarray(x)
        Z = (x_np - self.x_mean)/self.x_std
        return Z    

    def fit_transform(self, x):
        self.data = pd.DataFrame(x)
        self.x_std = np.asarray(self.data.std())
        self.x_mean = np.asarray(self.data.mean())
        Z = (self.data - self.data.mean())/self.data.std()
        return Z
    

class min_max:
    x_min = None
    x_max = None
    data = None
    
    def fit(self, x):
        self.data = pd.DataFrame(x)
        self.x_min = np.asarray(self.data.min())
        self.x_max = np.asarray(self.data.max())
    
    def transform(self, x):
        x_np = np.asarray(x)
        MM = (x_np - self.x_min)/(self.x_max - self.x_min)
        return MM
    
    def fit_transform(self, x):
        self.data = pd.DataFrame(x)
        self.x_min = np.asarray(self.data.min())
        self.x_max = np.asarray(self.data.max())
        MM = (self.data - self.data.min())/(self.data.max() - self.data.min())
        return MM