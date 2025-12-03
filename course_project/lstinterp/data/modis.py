"""MODIS datasets for training and testing"""
from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


@dataclass
class MODISConfig:
    """MODIS dataset configuration for training and testing"""
    data_path: str = "modis_aug_data/MODIS_Aug.mat" # path to the MODIS data
    split: Literal["train", "test"] = "train"
    normalize: bool = True
    lat_range: tuple = (35, 40)  # latitude range
    lon_range: tuple = (-115, -105)  # longitude range


class MODISDataset(Dataset):
    """
    Convert the 100x200x31 tensor to (N, 3) input + (N,) output
    or (T, H, W) image format, for CNN usage
    """
    def __init__(
        self,
        tensor: np.ndarray,
        mode: Literal["point", "image"] = "point",
        normalize_coords: bool = True,
        lat_range: tuple = (35, 40),
        lon_range: tuple = (-115, -105),
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
    ):
        """
        norm_mean, norm_std: if provided, use these values for normalization (for testing using the training set statistics)
        """
        self.mode = mode
        self.tensor = tensor.astype(np.float32)
        self.normalize_coords = normalize_coords
        self.lat_range = lat_range
        self.lon_range = lon_range

        self.lat_size, self.lon_size, self.time_size = self.tensor.shape

        # 0 represents missing, convert to NaN
        self.tensor[self.tensor == 0.0] = np.nan
        self.mask = ~np.isnan(self.tensor)

        if self.mode == "point":
            # extract the coordinates and values of all observed points
            lat_idx, lon_idx, t_idx = np.where(self.mask)
            
            # convert the indices to real coordinates
            lat_coords = np.linspace(lat_range[0], lat_range[1], self.lat_size)
            lon_coords = np.linspace(lon_range[0], lon_range[1], self.lon_size)
            
            self.coords = np.stack([
                lat_coords[lat_idx],
                lon_coords[lon_idx],
                t_idx.astype(np.float32)  # Time index
            ], axis=1)
            
            if normalize_coords:
                # normalize the coordinates
                self.coords[:, 0] = (self.coords[:, 0] - lat_range[0]) / (lat_range[1] - lat_range[0])
                self.coords[:, 1] = (self.coords[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0])
                self.coords[:, 2] = self.coords[:, 2] / (self.time_size - 1)
            
            self.values = self.tensor[self.mask]
            
        elif self.mode == "image":
            # (T, 1, H, W) - time dimension first
            self.images = np.moveaxis(self.tensor, 2, 0)[:, None, :, :]
            self.masks = np.moveaxis(self.mask.astype(np.float32), 2, 0)[:, None, :, :]
            
            # normalize: use provided statistics, or calculate the statistics of the current data
            if norm_mean is not None and norm_std is not None:
                # use provided statistics (training set)
                self.mean_val = norm_mean
                self.std_val = norm_std
            else:
                # use the statistics of the current data
                self.mean_val = np.nanmean(self.tensor)
                self.std_val = np.nanstd(self.tensor)
            
            # use the mean to fill the missing values (for input)
            self.images_filled = np.nan_to_num(self.images, nan=self.mean_val)
            # normalize the data (mean=0, std=1)
            self.images_normalized = (self.images_filled - self.mean_val) / (self.std_val + 1e-8)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        if self.mode == "point":
            return len(self.values)
        else:
            return self.images.shape[0]

    def __getitem__(self, idx):
        if self.mode == "point":
            x = self.coords[idx]
            y = self.values[idx]
            return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)
        else:
            # return the image and mask
            img = self.images_normalized[idx]
            mask = self.masks[idx]
            # if there is a true value, also normalize it (for training)
            if hasattr(self, 'images'):
                target_raw = self.images[idx]
                # normalize the target value
                target = (np.nan_to_num(target_raw, nan=self.mean_val) - self.mean_val) / (self.std_val + 1e-8)
            else:
                target = img
            return (
                torch.from_numpy(img).float(),
                torch.from_numpy(mask).float(),
                torch.from_numpy(target).float()
            )


def load_modis_tensor(mat_path: str, key: str = "training_tensor") -> np.ndarray:
    """
    Load MODIS data from .mat file and return the tensor
    """
    data = sio.loadmat(mat_path)
    tensor = data[key]  # shape: (100, 200, 31)
    return tensor


def create_spatial_temporal_coords(
    lat_size: int = 100,
    lon_size: int = 200,
    time_size: int = 31,
    lat_range: tuple = (35, 40),
    lon_range: tuple = (-115, -105),
    normalize: bool = True
) -> np.ndarray:
    """
    Create the spatial-temporal coordinate grid
    Returns: coordinates: (lat_size * lon_size * time_size, 3) - (lat, lon, time)
    """
    lat_coords = np.linspace(lat_range[0], lat_range[1], lat_size)
    lon_coords = np.linspace(lon_range[0], lon_range[1], lon_size)
    time_coords = np.arange(time_size)
    
    lat_grid, lon_grid, time_grid = np.meshgrid(
        lat_coords, lon_coords, time_coords, indexing='ij'
    )
    
    coords = np.stack([
        lat_grid.ravel(),
        lon_grid.ravel(),
        time_grid.ravel()
    ], axis=1)
    
    if normalize:
        coords[:, 0] = (coords[:, 0] - lat_range[0]) / (lat_range[1] - lat_range[0])
        coords[:, 1] = (coords[:, 1] - lon_range[0]) / (lon_range[1] - lon_range[0])
        coords[:, 2] = coords[:, 2] / (time_size - 1)
    
    return coords

