import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.filters import window
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.ndimage import shift, center_of_mass, zoom
from scipy.optimize import curve_fit
from scipy.fft import fftn, ifftn, fftshift
from numba import njit, prange, objmode
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import os

def fourier_upsample_4d(arr: np.ndarray, upsampling: int, worker: int | None = None) -> np.ndarray:
    if worker is None:
        worker = -1
    original_shape = arr.shape
    new_shape = original_shape[0], original_shape[1], original_shape[2] * upsampling, original_shape[3] * upsampling

    freq = fftshift(fftn(arr, axes=(2,3), workers=worker), axes=(2,3))

    # Pad freq with 0 to new shape
    pad0 = (new_shape[2] - original_shape[2]) // 2
    pad1 = new_shape[2] - original_shape[2] - pad0
    pad2 = (new_shape[3] - original_shape[3]) // 2
    pad3 = new_shape[3] - original_shape[3] - pad2

    upsampled_fft = np.pad(freq, ((0,0), (0,0), (pad0, pad1), (pad2, pad3)), mode='constant', constant_values=0)

    upsampled = ifftn(fftshift(upsampled_fft, axes=(2,3)), axes=(2,3), workers=worker)
    upsampled *= upsampling ** 2  # Scale the result

    return np.real(upsampled)


class ta4D_datasets:
    def __init__(self, data: list[np.ndarray]):
        """data: List of 4D data array to be processed.
        """
        self.N_scans = len(data)
        self.datasets = data
        self.datasets_cropped = data[:]  # Initialize cropped datasets as original
        self.shifts = np.zeros((self.N_scans, 2), dtype=np.int16)

    def filter_nan(self) -> None:
        """Filter out NaN values in the datasets by replacing them with zeros.
        """
        for i in range(self.N_scans):
            dataset = self.datasets[i]
            if np.isnan(dataset).any():
                print(f'NaN values found in dataset {i+1}, replacing with zeros.')
                dataset = np.nan_to_num(dataset, nan=0.0)
                self.datasets[i] = dataset

    @staticmethod
    def get_virtual_image_single(data, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None) -> np.ndarray:
        """Get virtual image from a single ta4D dataset for real-space alignment.
        type: Type of virtual image to generate. Options are 'brightfield' or 'darkfield'
        center: Center of the virtual image aperture. If None, will use the center of the dataset
        radius: Radius of the virtual image aperture. For 'brightfield', this is the radius of the brightfield disk. 
        For 'darkfield', this is the inner radius of the darkfield annulus. If None, for 'brightfield', will use 2 px; 
        for 'darkfield', will integrate from 1/2 of the maximum radius to the maximum radius.
        Returns:
        virtual_image: 2D numpy array of virtual image.
        """
        if center is None:
            center = (data.shape[2]//2, data.shape[3]//2)
        # Generate virtual mask according to type (diffraction-plane shape)
        virtual_mask = np.zeros((data.shape[2], data.shape[3]), dtype=bool)
        Y, X = np.ogrid[:data.shape[2], :data.shape[3]]
        dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        if type == 'brightfield':
            if radius is None:
                radius = 2  # Default brightfield radius
            virtual_mask[dist_from_center <= radius] = True
        elif type == 'darkfield':
            if radius is None:
                radius = dist_from_center.max() / 2  # Default darkfield inner radius
            virtual_mask[(dist_from_center >= radius) & (dist_from_center <= dist_from_center.max())] = True
        else:
            raise ValueError("Invalid type. Options are 'brightfield' or 'darkfield'.")
        
        # Generate virtual image
        # data shape expected (Ny, Nx, Qy, Qx); virtual_mask (Qy, Qx) broadcasts.
        virtual_image = np.sum(data * virtual_mask, axis=(2, 3))

        return virtual_image



    def get_virtual_images(self, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, plot_imgs: bool = True, *args, **kwargs) -> list[np.ndarray]:
        """Get virtual images from all ta4D datasets for real-space alignment.
        type: Type of virtual image to generate. Options are 'brightfield' or 'darkfield'
        center: Center of the virtual image aperture. If None, will use the center of the dataset
        radius: Radius of the virtual image aperture. For 'brightfield', this is the radius of the brightfield disk. 
        For 'darkfield', this is the inner radius of the darkfield annulus. If None, for 'brightfield', will use 2 px; 
        for 'darkfield', will integrate from 1/2 of the maximum radius to the maximum radius.
        plot_imgs: Whether to plot the virtual images.
        *args, **kwargs: Additional arguments passed to matplotlib.pyplot.imshow for plotting.
        Returns:
        virtual_images: List of 2D numpy arrays of virtual images.
        """
        
        # Generate virtual images
        self.virtual_images = []
        for i in tqdm(range(self.N_scans), desc="Generating virtual images"):
            dataset = self.datasets[i]
            virtual_img = self.get_virtual_image_single(dataset, type=type, center=center, radius=radius, plot_imgs=False)
            self.virtual_images.append(virtual_img)

        if plot_imgs:
            self.plot_virtual_images(*args, **kwargs)
        return self.virtual_images
    
    def plot_virtual_images(self, mask: np.ndarray | None = None, pvmin: float = 0, pvmax: float = 100, *args, **kwargs) -> None:
        '''Plot the virtual images stored in self.virtual_images.
        mask: Optional mask to overlay on the images. If provided, should be a 2D numpy array of the same shape as the virtual images.
        pvmin: Minimum value in percentile for imshow.
        pvmax: Maximum value in percentile for imshow.
        *args, **kwargs: Additional arguments passed to matplotlib.pyplot.imshow for plotting.
        '''
        assert hasattr(self, 'virtual_images'), "virtual_images not found. Please run get_virtual_images() first."
        if mask is not None:
            assert mask.shape == self.virtual_images[0].shape, "Mask shape does not match virtual image shape."
        else:
            mask = np.ones(self.virtual_images[0].shape)
        n_cols = 4
        n_rows = int(np.ceil(self.N_scans / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        for i, img in enumerate(self.virtual_images):
            axes.flat[i].imshow(img * mask, vmin=np.percentile(img, pvmin), vmax=np.percentile(img, pvmax), *args, **kwargs)
            axes.flat[i].set_title(f'Scan {i+1}')
            axes.flat[i].axis('off')
        # Turn off unused subplots
        if i + 1 < n_rows * n_cols:
            for j in range(i+1, n_rows*n_cols):
                axes.flat[j].axis('off')
        plt.tight_layout()
        plt.show()

    def get_mean_dp(self, dataset) -> np.ndarray:
        '''Get the mean diffraction pattern across 4d dataset.
        Returns:
        mean_dp: 2D numpy array of the mean diffraction pattern.
        '''
        mean_dp = np.mean(dataset, axis=(0, 1))
        return mean_dp
    
    def get_mean_dp_all_scans(self, plot_dps: bool = True, *args, **kwargs) -> list[np.ndarray]:
        '''Get the mean diffraction pattern for all scans.
        plot_dps: Whether to plot the mean diffraction patterns.
        *args, **kwargs: Additional arguments passed to plot_dps() for plotting.
        Returns:
        mean_dps: List of 2D numpy arrays of the mean diffraction patterns for each scan.
        '''
        self.mean_dps = []
        for i in tqdm(range(self.N_scans), desc="Calculating mean diffraction patterns"):
            dataset = self.datasets[i]
            mean_dp = self.get_mean_dp(dataset)
            self.mean_dps.append(mean_dp)
        if plot_dps:
            self.plot_dps(*args, **kwargs)
        return self.mean_dps
    
    def plot_dps(self, mask: np.ndarray | None = None, pvmin: float = 0, pvmax: float = 100, gamma: float = 1, *args, **kwargs) -> None:
        '''Plot the mean diffraction patterns stored in self.mean_dps.
        mask: Optional mask to overlay on the diffraction patterns. If provided, should be a 2D numpy array of the same shape as the diffraction patterns.
        pvmin: Minimum value in percentile for imshow.
        pvmax: Maximum value in percentile for imshow.
        gamma: Gamma correction factor.
        *args, **kwargs: Additional arguments passed to matplotlib.pyplot.imshow for plotting.
        '''
        assert hasattr(self, 'mean_dps'), "mean_dps not found. Please run get_mean_dp_all_scans() first."
        if mask is not None:
            assert mask.shape == self.mean_dps[0].shape, "Mask shape does not match diffraction pattern shape."
        else:
            mask = np.ones(self.mean_dps[0].shape)
        n_cols = 4
        n_rows = int(np.ceil(self.N_scans / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        for i, dp in enumerate(self.mean_dps):
            vmin = np.percentile(dp, pvmin)
            vmax = np.percentile(dp, pvmax)
            axes.flat[i].imshow(dp * mask, norm=colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax), *args, **kwargs)
            axes.flat[i].set_title(f'Scan {i+1} Mean DP')
            axes.flat[i].axis('off')
        # Turn off unused subplots
        if i + 1 < n_rows * n_cols:
            for j in range(i+1, n_rows*n_cols):
                axes.flat[j].axis('off')
        plt.tight_layout()
        plt.show()
    
    def find_img_shift_virtual_imgs(self, imgs: list[np.ndarray] | None = None, apply_hann_window: bool =False, print_shifts: bool =True, **kwargs) -> np.ndarray:
        '''Find the image shift for all scans using a virtual image signal
        imgs: sequence of 2D numpy array of virtual images. If None, will use self.virtual_images
        apply_hann_window: bool, whether to apply a hann window before alignment
        print_shifts: bool, whether to print the shifts for each scan
        **kwargs: other keyword arguments to be passed into skimage.registration.phase_cross_correlation
        Note: the shifts should be integer pixles. Do not pass upsample_factor in kwargs.
        Returns:
        shifts: numpy array of shape (N_imgs, 2), the shifts for each image
        '''
        if imgs is None:
            imgs = self.virtual_images
        N_imgs = len(imgs)
        data_shape = imgs[0].shape
        # shifts = np.zeros((N_imgs, 2), dtype=np.int16)
        
        for i in tqdm(range(1, N_imgs), desc="Finding image shifts"):
            reference_img = imgs[i-1]
            moving_img = imgs[i]
            if apply_hann_window:
                hann_window = window('hann', data_shape)
                reference_img = reference_img * hann_window
                moving_img = moving_img * hann_window
            shift_i, _, _ = phase_cross_correlation(reference_img, moving_img, **kwargs)
            self.shifts[i] = shift_i + self.shifts[i-1]  # Accumulate shifts
            if print_shifts:
                print(f'Shift of scan {i+1}: (y, x) = ({self.shifts[i,0]}, {self.shifts[i,1]})')
        return self.shifts
    
    def align_scans_with_shifts(self, shifts: np.ndarray, crop_virtual_imgs: bool = True, print_cropping_info: bool = True) -> None:
        '''Align the scans using the provided shifts
        shifts: numpy array of shape (N_scans, 2), the shifts for each scan, must be integers
        print_cropping_info: bool, whether to print the cropping information for each scan
        '''  
        self.datasets_cropped = self.datasets[:]  
        y_shifts = shifts[:, 0]
        x_shifts = shifts[:, 1]
        data_shape = self.datasets[0].shape[:2]  
        # Determine the cropping region
        y_min = max(0, max(y_shifts))
        y_max = min(data_shape[0], data_shape[0] + min(y_shifts))
        x_min = max(0, max(x_shifts))
        x_max = min(data_shape[1], data_shape[1] + min(x_shifts))
        # Apply the shifts and cropping to all datasets
        for i in tqdm(range(self.N_scans), desc="Aligning scans"):
            y_shift = y_shifts[i]
            x_shift = x_shifts[i]
            y0 = y_min - y_shift
            y1 = y_max - y_shift
            x0 = x_min - x_shift
            x1 = x_max - x_shift
            self.datasets_cropped[i] = self.datasets[i][y0:y1, x0:x1, :, :]
            if crop_virtual_imgs and hasattr(self, 'virtual_images'):
                self.virtual_images[i] = self.virtual_images[i][y0:y1, x0:x1]
            if print_cropping_info:
                print(f'Scan {i+1} cropped from original indices of {y0}:{y1}, {x0}:{x1}, yielding a shape: {self.datasets_cropped[i].shape}')

    

    @staticmethod
    def get_origin_single_dp(dp: np.ndarray, center_guess: tuple | None = None, refine_radius: int = 5, threshold: float = 1.5) -> tuple:
        '''Get the origin of a single diffraction pattern using center of mass method.
        dp: 2D numpy array of the diffraction pattern
        center_guess: Tuple of (x, y) coordinates for initial center guess. If None, will use the maximum pixel.
        refine_radius: int, radius around the center_guess to consider for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation
        Returns:
        origin: Tuple of (x, y) coordinates of the origin
        '''
        nqy, nqx = dp.shape
        if center_guess is None:
            center_guess = np.unravel_index(np.argmax(dp), dp.shape)
        y0 = max(0, round(center_guess[1] - refine_radius))
        y1 = min(nqy, round(center_guess[1] + refine_radius + 1))
        x0 = max(0, round(center_guess[0] - refine_radius))
        x1 = min(nqx, round(center_guess[0] + refine_radius + 1))
        dp_crop = dp[y0:y1, x0:x1]
        # Apply threshold
        binary_thresh = np.mean(dp_crop) + threshold * np.std(dp_crop)
        binary_dp = dp_crop > binary_thresh
        
        # Calculate CoM 
        com_y, com_x = center_of_mass(binary_dp)
        com_y += y0
        com_x += x0
        
        return (com_x, com_y)
    

    @staticmethod
    def hough_transform_circle(data: np.ndarray, radius: np.ndarray, sigma: float = 0.5, low_threshold: float = None, high_threshold: float = None) -> tuple:
        '''Detect circle in 2D data using Hough Transform.
        data: 2D numpy array of the image to be processed
        radius: list or 1D numpy array of the radius of the circle to be detected
        sigma: float, standard deviation for the Gaussian filter used in Canny edge detection
        low_threshold: float, low threshold for Canny edge detection. If None, will be set to 10% of max value.
        high_threshold: float, high threshold for Canny edge detection. If None, will be set to 20% of max value.
        Returns:
        center: Tuple of (x, y) coordinates and radius of the detected circle center
        '''
        

        edges = canny(data, sigma=sigma)
        hough_radii = np.array(radius)
        hough_res = hough_circle(edges, hough_radii)
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        return cx, cy, radii


    

    def align_scan_positions(self, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, 
                             apply_hann_window: bool =False, print_shifts: bool =True, print_cropping_info: bool = True, **kwargs) -> None:
        '''High-level function to align all scans in the ta4D dataset.
        **kwargs: Additional arguments passed to skimage.registration.phase_cross_correlation.'''
        max_workers = min(os.cpu_count() or 1, self.N_scans)
        self.get_virtual_images_parallel(max_workers=max_workers, type=type, center=center, radius=radius)
        self.find_img_shift_virtual_imgs(apply_hann_window=apply_hann_window, print_shifts=print_shifts, **kwargs)
        self.align_scans_with_shifts(self.shifts, print_cropping_info=print_cropping_info)


    def get_dp_origins(self, center_guess: tuple | None = None, refine_radius: int = 5, threshold: float = 1.5) -> list[np.ndarray]:
        '''Get the origins of the diffraction pattern for all scans.
        Will call get_origin_single_dp for each diffraction pattern.'''
        self.dp_origins = []
        for i in range(self.N_scans):
            dataset = self.datasets_cropped[i]
            ny, nx, _, _ = dataset.shape
            origins_scan = np.zeros((ny, nx, 2))
            total_dps = ny * nx
            with tqdm(total=total_dps, desc=f'Processing scan {i+1}') as pbar:
                for iy in range(ny):
                    for ix in range(nx):
                        dp = dataset[iy, ix]
                        origin = self.get_origin_single_dp(dp, center_guess=center_guess, refine_radius=refine_radius, threshold=threshold)
                        origins_scan[iy, ix] = origin
                        pbar.update(1)
            self.dp_origins.append(origins_scan)
        return self.dp_origins
    
    @staticmethod
    @njit(parallel=True)
    def get_dp_origins_single_scan(dataset: np.ndarray, center_guess: tuple | None = None, refine_radius: int = 5, threshold: float = 1.5) -> np.ndarray:
        '''Get the origins of the diffraction patterns for a single scan. Use numba for acceleration.
        data: Tuple containing a 4D numpy array of the dataset to be processed
        center_guess: Tuple of (x, y) coordinates for initial center guess. If None, will use the center of the array.
        refine_radius: int, radius around the center_guess to consider for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation.
        Returns:
        origins_scan: 3D numpy array of shape (Ny, Nx, 2) of the origins for each diffraction pattern
        '''
        ny, nx, _, _ = dataset.shape
        origins_scan = np.zeros((ny, nx, 2))
        for iy in prange(ny):
            for ix in range(nx):
                dp = dataset[iy, ix]
                nqy, nqx = dp.shape
                if center_guess is None:
                    center_x, center_y = nqx // 2, nqy // 2
                else:
                    center_x, center_y = center_guess
                y0 = max(0, round(center_y - refine_radius))
                y1 = min(nqy, round(center_y + refine_radius + 1))
                x0 = max(0, round(center_x - refine_radius))
                x1 = min(nqx, round(center_x + refine_radius + 1))
                dp_crop = dp[y0:y1, x0:x1]

                binary_thresh = np.mean(dp_crop) + threshold * np.std(dp_crop)
                binary_dp = dp_crop > binary_thresh
                
                # Calculate CoM - manual loop 
                total_intensity = np.sum(binary_dp)
                dp_shape_y, dp_shape_x = binary_dp.shape
                if total_intensity == 0:
                    com_y = dp_shape_y // 2
                    com_x = dp_shape_x // 2
                else:
                    sum_y = 0.0
                    sum_x = 0.0
                    for yy in range(dp_shape_y):
                        for xx in range(dp_shape_x):
                            if binary_dp[yy, xx]:
                                sum_y += yy
                                sum_x += xx
                    com_y = sum_y / total_intensity
                    com_x = sum_x / total_intensity

                com_y += y0
                com_x += x0
                
                origins_scan[iy, ix, 0] = com_x
                origins_scan[iy, ix, 1] = com_y

        return origins_scan

    
    def get_dp_origins_numba(self, center_guess: tuple | None = None, refine_radius: int = 5, threshold: float = 1.5) -> list[np.ndarray]:
        '''Get the origins of the diffraction pattern for all scans using numba for acceleration.'''
        self.dp_origins = []
        for i in tqdm(range(self.N_scans), desc="Getting dp origins"):
            dataset = self.datasets_cropped[i]
            origins_scan = self.get_dp_origins_single_scan(dataset, center_guess=center_guess, refine_radius=refine_radius, threshold=threshold)
            self.dp_origins.append(origins_scan)


    def fit_origins_plane(self, plot_fit: bool = False) -> list[np.ndarray]:
        '''Fit a plane to the dp origins for each scan to account for systematic shifts induced by TEM.
        plot_fit: bool, whether to plot the fitted plane against the original origins
        Returns:
        fitted_origins: List of 3D numpy arrays of shape (Ny, Nx, 2) for each scan
        '''
        assert hasattr(self, 'dp_origins'), "dp_origins not found. Please run get_dp_origins() first."
        self.fitted_dp_origins = []
        # Prepare data for fitting
        ny, nx, _ = self.dp_origins[0].shape
        Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        xdata = np.vstack((X.ravel(), Y.ravel()))
        for i in tqdm(range(self.N_scans), desc="Fitting origins plane"):
            origins_scan = self.dp_origins[i]
            Z_x = origins_scan[:, :, 0]  # x origins
            Z_y = origins_scan[:, :, 1]  # y origins
            # Fit plane for x origins
            Z = Z_x.ravel()
            A = np.c_[xdata.T, np.ones(xdata.shape[1])]
            C_x, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            # Fit plane for y origins
            Z = Z_y.ravel()
            C_y, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            # Generate fitted origins
            Z_x_fit = (C_x[0] * X + C_x[1] * Y + C_x[2]).reshape(ny, nx)
            Z_y_fit = (C_y[0] * X + C_y[1] * Y + C_y[2]).reshape(ny, nx)
            fitted_origins_scan = np.zeros((ny, nx, 2))
            fitted_origins_scan[:, :, 0] = Z_x_fit
            fitted_origins_scan[:, :, 1] = Z_y_fit
            self.fitted_dp_origins.append(fitted_origins_scan)
            if plot_fit:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                im0 = axes[0].imshow(origins_scan[:, :, 0], cmap='viridis')
                axes[0].set_title(f'Scan {i+1} Original X Origins')
                fig.colorbar(im0, ax=axes[0])
                im1 = axes[1].imshow(Z_x_fit, cmap='viridis')
                axes[1].set_title(f'Scan {i+1} Fitted X Origins')
                fig.colorbar(im1, ax=axes[1])
                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                im0 = axes[0].imshow(origins_scan[:, :, 1], cmap='viridis')
                axes[0].set_title(f'Scan {i+1} Original Y Origins')
                fig.colorbar(im0, ax=axes[0])
                im1 = axes[1].imshow(Z_y_fit, cmap='viridis')
                axes[1].set_title(f'Scan {i+1} Fitted Y Origins')
                fig.colorbar(im1, ax=axes[1])
                plt.show()
        return self.fitted_dp_origins
    
    @staticmethod
    def align_dps_single_scan(data: tuple, upsampling: int = 1, tqdm_desc: str | None = None) -> np.ndarray:
        '''Align the diffraction patterns in a single 4D dataset using the provided origins.
        data: Tuple containing a 4D numpy array of the dataset to be aligned and a 3D numpy array of shape (Ny, Nx, 2) of the origins for each diffraction pattern      
        upsampling: Factor by which to upsample the diffraction patterns before alignment
        tqdm_desc: Optional description for the tqdm progress bar
        Returns:
        aligned_dataset: 4D numpy array of the aligned dataset
        '''
        if tqdm_desc is None:
            tqdm_desc = "Aligning DPs"
        dataset, dp_origins = data
        ny, nx, _, _ = dataset.shape
        if upsampling > 1:
            # Upsample dataset
            # dataset = zoom(dataset, (1, 1, upsampling, upsampling))
            dataset = fourier_upsample_4d(dataset, upsampling)
            dp_origins = dp_origins * upsampling
        aligned_dataset = np.zeros_like(dataset)
        total_dps = ny * nx
        with tqdm(total=total_dps, desc=tqdm_desc) as pbar:
            for iy in range(ny):
                for ix in range(nx):
                    dp = dataset[iy, ix]
                    origin = dp_origins[iy, ix]
                    offset_y = (dp.shape[0] / 2) - origin[1]
                    offset_x = (dp.shape[1] / 2) - origin[0]
                    shifted = shift(dp, (offset_y, offset_x))
                    aligned_dataset[iy, ix] = shifted
                    pbar.update(1)
        return aligned_dataset
    
    @staticmethod
    @njit(parallel=True)
    def align_dps_single_scan_parallel(data: tuple) -> np.ndarray:
        '''Align the diffraction patterns in a single 4D dataset using the provided origins in parallel.
        data: Tuple containing a 4D numpy array of the dataset to be aligned and a 3D numpy array of shape (Ny, Nx, 2) of the origins for each diffraction pattern
        Returns:
        aligned_dataset: 4D numpy array of the aligned dataset
        '''
        dataset, dp_origins = data
        dataset = np.ascontiguousarray(dataset)
        dp_origins = np.ascontiguousarray(dp_origins)
        ny, nx, _, _ = dataset.shape
        aligned_dataset = np.zeros(dataset.shape, dtype=dataset.dtype)
        
        for iy in prange(ny):
            for ix in range(nx):
                dp = dataset[iy, ix]
                origin = dp_origins[iy, ix]
                offset_y = (dp.shape[0] / 2) - origin[1]
                offset_x = (dp.shape[1] / 2) - origin[0]
                with objmode(shifted='float32[:,:]'):
                    shifted = shift(dp, (offset_y, offset_x))
                aligned_dataset[iy, ix] = shifted
        
        return aligned_dataset


    def align_dps(self, center_guess: tuple | None = None, refine_radius: int = 5, threshold: float = 1.5, fit_plane: bool = True) -> np.ndarray:
        '''High-level function to align all diffraction patterns in all scans.
        First get the dp origins using get_dp_origins, then align using the origins.
        center_guess: Tuple of (x, y) coordinates for initial center guess. If None, will use the maximum pixel.
        refine_radius: int, radius around the center_guess to consider for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation
        fit_plane: bool, whether to fit a plane to the origins before alignment
        '''
        assert hasattr(self, 'datasets_cropped'), "datasets_cropped not found. Please run align_scan_positions() first."
        self.get_dp_origins(center_guess=center_guess, refine_radius=refine_radius, threshold=threshold)
        if fit_plane:
            self.fit_origins_plane()
            origins_to_use = self.fitted_dp_origins
        else:
            origins_to_use = self.dp_origins

        self.aligned_datasets = np.zeros(self.datasets_cropped[0].shape, dtype=self.datasets_cropped[0].dtype)
        for i in range(self.N_scans):
            dataset = self.datasets_cropped[i]
            origins_scan = origins_to_use[i]
            aligned_dataset = self.align_dps_single_scan((dataset, origins_scan), tqdm_desc=f'Aligning scan {i+1}')
            self.aligned_datasets += aligned_dataset
        return self.aligned_datasets
    

    def run_parallel(self, func, iterable, max_workers: int = None, tqdm_desc: str = "Running in parallel", *args, **kwargs) -> list:
        '''Run a function in parallel over an iterable using ProcessPoolExecutor.
        func: Function to be executed in parallel.
        iterable: Iterable of inputs to the function.
        max_workers: Maximum number of worker processes to use. If None, will use the number of processors on the machine.
        tqdm_desc: Description for the tqdm progress bar.
        *args, **kwargs: Additional arguments to pass to the function.
        Returns:
        results: List of results from the function calls.
        '''
        if max_workers is None:
            max_workers = min(len(iterable), os.cpu_count() or 1)
        
        # Use partial to bind kwargs to the function
        func_with_kwargs = partial(func, *args, **kwargs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(func_with_kwargs, iterable), total=len(iterable), desc=tqdm_desc))
        return results
    
    def get_virtual_images_parallel(self, max_workers: int = None, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None) -> list[np.ndarray]:
        '''Get virtual images from all ta4D datasets in parallel for real-space alignment.
        Uses ThreadPoolExecutor instead of ProcessPoolExecutor for better performance with numpy operations.'''
        self.virtual_images = self.run_parallel(
            func=self.get_virtual_image_single,
            iterable=self.datasets,
            max_workers=max_workers,
            tqdm_desc="Generating virtual images",
            type=type,
            center=center,
            radius=radius,
        )

    def align_dps_parallel(self, dp_origins: list, max_workers: int = None) -> np.ndarray:
        '''Align all diffraction patterns in all scans in parallel.
        Uses ProcessPoolExecutor for CPU-bound work with numpy operations.'''
        if max_workers is None:
            max_workers = min(self.N_scans, os.cpu_count() or 1)
        
        # Prepare data tuples
        data_tuples = [(self.datasets_cropped[i], dp_origins[i], i+1) for i in range(self.N_scans)]
        
        self.aligned_datasets = np.zeros(self.datasets_cropped[0].shape, dtype=self.datasets_cropped[0].dtype)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.align_dps_single_scan, 
                                     (data_tuples[i][0], data_tuples[i][1]), 
                                     tqdm_desc=f"Aligning scan {data_tuples[i][2]}")
                      for i in range(len(data_tuples))]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning diffraction patterns"):
                self.aligned_datasets += future.result()

        return self.aligned_datasets
    
    def align_dps_numba(self, dp_origins: list, upsampling: int = 1) -> np.ndarray:
        '''Align all diffraction patterns in all scans using numba for acceleration.
        Returns:
        aligned_datasets: 4D numpy array of the summed aligned datasets.
        '''
        assert hasattr(self, 'datasets_cropped'), "datasets_cropped not found. Please run align_scan_positions() first."
        aligned_shape = self.datasets_cropped[0].shape
        if upsampling > 1:
            aligned_shape = (aligned_shape[0], aligned_shape[1], aligned_shape[2]*upsampling, aligned_shape[3]*upsampling)
        self.aligned_datasets = np.zeros(aligned_shape, dtype=self.datasets_cropped[0].dtype)
        pbar = tqdm(range(self.N_scans), desc="Aligning DPs")
        for i in pbar:
            pbar.set_postfix({'scan': f'{i+1}/{self.N_scans}'})
            dataset = self.datasets_cropped[i]
            origins_scan = dp_origins[i]
            if upsampling > 1:
                # Upsample dataset
                # dataset = zoom(dataset, (1, 1, upsampling, upsampling))
                dataset = fourier_upsample_4d(dataset, upsampling)
                origins_scan = origins_scan * upsampling
            aligned_dataset = self.align_dps_single_scan_parallel((dataset, origins_scan))
            self.aligned_datasets += aligned_dataset
        return self.aligned_datasets
    

    def align_scans_dps(self, 
                        type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, apply_hann_window: bool =False,
                        refine_radius: int = 5, threshold: float = 1.5, fit_plane: bool = True, upsampling: int = 1,
                        use_numba: bool = False) -> np.ndarray:
        '''High-level function to align all diffraction patterns in all scans.
        First get the dp origins using get_dp_origins, then align using the origins.
        center: Tuple of (x, y) coordinates for virtual images and initial center guess. If None, will use the center of the image array.
        refine_radius: int, radius around the center to consider for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation
        fit_plane: bool, whether to fit a plane to the origins before alignment
        upsampling: Factor by which to upsample the diffraction patterns before alignment
        use_numba: bool, whether to use numba for acceleration
        '''
        # First, align the scans in real space
        self.align_scan_positions(type=type, center=center, radius=radius, apply_hann_window=apply_hann_window)
        if use_numba:
            self.get_dp_origins_numba(center_guess=center, refine_radius=refine_radius, threshold=threshold)
        else:
            self.get_dp_origins(center_guess=center, refine_radius=refine_radius, threshold=threshold)
        if fit_plane:
            self.fit_origins_plane()
            origins_to_use = self.fitted_dp_origins
        else:
            origins_to_use = self.dp_origins

        if use_numba:
            self.aligned_datasets = self.align_dps_numba(origins_to_use, upsampling=upsampling)
        else:
            # self.aligned_datasets = self.align_dps_parallel(origins_to_use)
            aligned_shape = self.datasets_cropped[0].shape
            if upsampling > 1:
                aligned_shape = (aligned_shape[0], aligned_shape[1], aligned_shape[2]*upsampling, aligned_shape[3]*upsampling)
            self.aligned_datasets = np.zeros(aligned_shape, dtype=self.datasets_cropped[0].dtype)
            for i in range(self.N_scans):
                dataset = self.datasets_cropped[i]
                origins_scan = origins_to_use[i]
                aligned_dataset = self.align_dps_single_scan((dataset, origins_scan), upsampling=upsampling, tqdm_desc=f'Aligning scan {i+1}')
                self.aligned_datasets += aligned_dataset
        
        return self.aligned_datasets


#================ Old functions not used anymore but kept for reference =================#    
        

    def align_scans_dps_old(self, method: str = 'CoM', mask: tuple | None = None, 
                        CoM_mask: tuple | None = None, CoM_threshold: float = 1.5, 
                        upsample_factor: int = 10, normalization: str | None = 'phase', 
                        print_progress: bool = True) -> np.ndarray:
        '''Align all scans in the ta4D dataset using diffraction pattern alignment.
        method: Alignment method, 'CoM' for center of mass, 'xC' for phase cross-correlation
        mask: For xC only. None or 4-tuple y0,y1,x0,x1 defining the crop region to use for alignment
        CoM_mask: None or 4-tuple defining the crop region for centering with CoM (for 'xC' this is used for the first dp)
        CoM_threshold: float, threshold to generate binary mask for CoM calculation
        upsample_factor: For xC only. Upsampling factor for subpixel accuracy in phase cross-correlation
        normalization: For xC only. 'phase' for phase correlation; None for cross-correlation
        print_progress: bool, whether to print progress information
        Returns:
        Summed aligned dataset
        '''
        self.aligned_data = np.zeros_like(self.datasets[0])
        for i in range(self.N_scans):
            if print_progress:
                print(f'Aligning scan {i+1}/{self.N_scans} using method {method}...')
            data = np.ascontiguousarray(self.datasets[i])
            if method == 'CoM':
                aligned_data= self.align_dps_CoM(data, mask=CoM_mask, threshold=CoM_threshold)
            elif method == 'xC':
                # Flatten to S-shape
                data_flatten = self.flatten_s_shape(data)
                aligned_flatten = self.align_dps_xC(data_flatten, mask=mask, CoM_mask=CoM_mask, 
                                                    CoM_threshold=CoM_threshold, upsample_factor=upsample_factor, 
                                                    normalization=normalization)
                aligned_data = self.reconstruct_s_shape(aligned_flatten, data.shape)
            else:
                raise ValueError("Invalid method. Options are 'CoM' or 'xC'.")
            self.aligned_data += aligned_data

        return self.aligned_data


    @staticmethod
    @njit(parallel=True)
    def align_dps_CoM(data: np.ndarray, mask: tuple | None = None, threshold: float =1.5) -> np.ndarray:
        '''Align the diffraction patterns in a 4D dataset using center of mass (CoM) method.
        data: numpy array of single scan 4D dataset to be aligned
        mask: None or 4-tuple y0,y1,x0,x1 defining the crop region to use for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation
        Returns:
        aligned_data: numpy array of aligned 4D dataset
        '''
        ny, nx, nqy, nqx = data.shape
        img_center_x = nqx // 2
        img_center_y = nqy // 2
        data_flatten = data.reshape(ny*nx, nqy, nqx)
        # Align the datasets one by one from the data_flatten
        for i in prange(len(data_flatten)):
            dp = data_flatten[i]
            # Apply mask and threshold
            if mask is not None:
                dp = dp[mask[0]:mask[1], mask[2]:mask[3]]
                dp_shape_y = mask[1] - mask[0]
                dp_shape_x = mask[3] - mask[2]
            else:
                dp_shape_y, dp_shape_x = dp.shape
            binary_thresh = np.mean(dp) + threshold * np.std(dp)
            binary_dp = dp > binary_thresh
            
            # Calculate CoM - manual loop instead of np.indices
            total_intensity = np.sum(binary_dp)
            if total_intensity == 0:
                com_y = dp_shape_y // 2
                com_x = dp_shape_x // 2
            else:
                sum_y = 0.0
                sum_x = 0.0
                for yy in range(dp_shape_y):
                    for xx in range(dp_shape_x):
                        if binary_dp[yy, xx]:
                            sum_y += yy
                            sum_x += xx
                com_y = sum_y / total_intensity
                com_x = sum_x / total_intensity
            if mask is not None:
                com_y += mask[0]
                com_x += mask[2]
            
            offset_y = img_center_y - com_y
            offset_x = img_center_x - com_x
            
            # Apply shift to the full diffraction pattern
            with objmode(shifted='float32[:,:]'):
                shifted = shift(data_flatten[i], (offset_y, offset_x))

            data_flatten[i] = shifted
        
        return data_flatten.reshape(ny, nx, nqy, nqx)
    
    @staticmethod
    @njit(parallel=True)
    def align_dps_xC(data: np.ndarray, mask: tuple | None = None, 
                     CoM_mask: tuple | None = None, CoM_threshold: float = 1.5, 
                     upsample_factor: int = 10, normalization: str | None = 'phase') -> np.ndarray:
        ''' Align the diffraction patterns in a 4D dataset using phase cross-correlation.
        The first diffraction pattern needs to be centered using CoM method.
        data: numpy array of s-flatten single scan 4D dataset to be aligned
        mask: None or 4-tuple y0,y1,x0,x1 defining the crop region to use for alignment
        CoM_mask: None or 4-tuple defining the crop refion for the first dp for centering with CoM
        upsample_factor: upsampling factor for subpixel accuracy in phase cross-correlation
        normalization: normalization method for phase cross-correlation'''
        n_points, nqy, nqx = data.shape
        img_center_x = nqx // 2
        img_center_y = nqy // 2
        shifts = np.zeros((n_points, 2))
        aligned = np.zeros(data.shape)
        # Find the first dp center with CoM
        first_dp = data[0]
        if CoM_mask is not None:
            first_dp_crop = first_dp[CoM_mask[0]:CoM_mask[1], CoM_mask[2]:CoM_mask[3]]
            binary_thresh = np.mean(first_dp_crop) + CoM_threshold * np.std(first_dp_crop)
            binary_dp = first_dp_crop > binary_thresh
            total_intensity = np.sum(binary_dp)
            if total_intensity == 0:
                com_y = (CoM_mask[0] + CoM_mask[1]) / 2
                com_x = (CoM_mask[2] + CoM_mask[3]) / 2
            else:
                sum_y = 0.0
                sum_x = 0.0
                for yy in range(CoM_mask[1] - CoM_mask[0]):
                    for xx in range(CoM_mask[3] - CoM_mask[2]):
                        if binary_dp[yy, xx]:
                            sum_y += yy
                            sum_x += xx
                com_y = sum_y / total_intensity + CoM_mask[0]
                com_x = sum_x / total_intensity + CoM_mask[2]
            offset_y = img_center_y - com_y
            offset_x = img_center_x - com_x
            # Shift the first dp to center
            with objmode(shifted='float32[:,:]'):
                shifted = shift(first_dp, (offset_y, offset_x))
            data[0] = shifted
        aligned[0] = data[0]
        for i in prange(data.shape[0]-1):
            ref = data[i]
            moving = data[i+1]
            if mask is not None:
                y0, y1, x0, x1 = mask
                ref = ref[y0:y1, x0:x1]
                moving = moving[y0:y1, x0:x1]

            # Calculate shift in obj mode    
            with objmode(offset='float32[:]'):
                offset = phase_cross_correlation(ref, moving, upsample_factor=upsample_factor, normalization=normalization)[0]

            shifts[i+1] = offset + shifts[i]

            # Align in obj mode
            with objmode(shifted='float32[:,:]'):
                shifted = shift(data[i+1], offset)

            aligned[i+1] = shifted

        return aligned


    @staticmethod
    def flatten_s_shape(arr: np.ndarray) -> np.ndarray:
        """
        Flatten 4D array (ny, nx, nqy, nqx) to (ny*nx, nqy, nqx) in S-shape
        """
        ny, nx, nqy, nqx = arr.shape
        
        # Create a copy and reverse every other row in the first dimension
        temp = arr.copy()
        temp[1::2] = temp[1::2, ::-1]  # Reverse every other ny row
        
        # Reshape to flatten the first two dimensions
        flattened = temp.reshape(ny * nx, nqy, nqx)
        return flattened

    @staticmethod
    def reconstruct_s_shape(flattened: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Reconstruct 4D array from S-shaped flattened array
        flattened: array of shape (ny*nx, nqy, nqx)
        original_shape: tuple (ny, nx, nqy, nqx)
        """
        ny, nx, nqy, nqx = original_shape
        
        # First reshape back to (ny, nx, nqy, nqx)
        reconstructed = flattened.reshape(ny, nx, nqy, nqx)
        
        # Reverse every other row back to original order
        reconstructed[1::2] = reconstructed[1::2, ::-1]
        
        return reconstructed
