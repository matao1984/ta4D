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
from matplotlib.widgets import Slider
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from joblib import Parallel, delayed
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

def ewpc2D(data: np.ndarray, useWindow: bool = True, minlog: float = 0.1) -> np.ndarray:
        '''
        Calculates the EWPC transform--fft(log(data))--for 2d data
            For the theoretical background, check Padgett et al., Ultramicroscopy 2020
            (https://doi.org/10.1016/j.ultramic.2020.112994).
            
        :Parameters:
            data : 2D array
                2d diffraction data, ordered (ky, kx). If nonsquare, data will be padded to square with zeros.
            useWindow : a boolean
                'True' applies a hanning window before the FFT. The window may 
                prevent FFT artifacts caused by non-periodic boundaries.
            minlog : float, optional
                A small constant added to the data before taking the log to prevent
                log(0) or log of negative numbers. Default is 0.1.

        :Return: 
            cep : 2D array
                ceptral transformed data
        '''
        # Pad the data to a square shape if it's not already square
        data_shape = 0, data.shape[0], 0, data.shape[1]
        if data.shape[0] != data.shape[1]:                
            max_dim = max(data.shape)
            min_dim = min(data.shape)
            pad_width = int((max_dim - min_dim) / 2)
            padded_data = np.zeros((max_dim, max_dim))
            if data.shape[0] < data.shape[1]:  # Pad height
                data_shape = pad_width, pad_width+data.shape[0], 0, data.shape[1]
            else:  # Pad width
                data_shape = 0, data.shape[0], pad_width, pad_width+data.shape[1]
            padded_data[data_shape[0]:data_shape[1], data_shape[2]:data_shape[3]] = data
            data = padded_data

        logdp = np.log(data - np.min(data) + minlog) #shifts the data to positive values for the log
        if useWindow:
            win = window('hann', data.shape)
            logdp *= win
        cep = np.abs(np.fft.fftshift(np.fft.fft2(logdp)))
        return cep[data_shape[0]:data_shape[1], data_shape[2]:data_shape[3]]  # Crop back to original shape if padded


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

    def flip_dps(self, vertical: bool = False, horizontal: bool = False) -> None:
        """Flip the diffraction patterns in the datasets vertically and/or horizontally.
        vertical: Whether to flip vertically (up-down).
        horizontal: Whether to flip horizontally (left-right).
        """
        for i in range(self.N_scans):
            dataset = self.datasets[i]
            if vertical:
                dataset = np.flip(dataset, axis=2)  # Flip along ky axis
            if horizontal:
                dataset = np.flip(dataset, axis=3)  # Flip along kx axis
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
            virtual_img = self.get_virtual_image_single(dataset, type=type, center=center, radius=radius)
            self.virtual_images.append(virtual_img)

        if plot_imgs:
            self.plot_virtual_images(*args, **kwargs)
        return self.virtual_images
    
    def plot_virtual_images(self, mask: np.ndarray | None = None, stack: bool = False, pvmin: float = 0, pvmax: float = 100, *args, **kwargs) -> None:
        '''Plot the virtual images stored in self.virtual_images.
        mask: Optional mask to overlay on the images. If provided, should be a 2D numpy array of the same shape as the virtual images.
        stack: Whether to stack the images together. If True, will plot one image with a slider. Otherwise, will plot all images in a grid.
        pvmin: Minimum value in percentile for imshow.
        pvmax: Maximum value in percentile for imshow.
        *args, **kwargs: Additional arguments passed to matplotlib.pyplot.imshow for plotting.
        '''
        assert hasattr(self, 'virtual_images'), "virtual_images not found. Please run get_virtual_images() first."
        if mask is not None:
            assert mask.shape == self.virtual_images[0].shape, "Mask shape does not match virtual image shape."
        else:
            mask = np.ones(self.virtual_images[0].shape)
        if not stack:
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
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            img_display = ax.imshow(self.virtual_images[0] * mask, vmin=np.percentile(self.virtual_images[0], pvmin), vmax=np.percentile(self.virtual_images[0], pvmax), *args, **kwargs)
            ax.set_title('Scan 1')
            slider_ax = fig.add_axes([0.1, 0.05, 0.9, 0.03])
            scan_slider = Slider(
                ax=slider_ax,
                label='Scan',
                valmin=1,
                valmax=self.N_scans,
                valinit=1,
                valstep=1,
                valfmt='%d'
            )
            def update(val):
                img_display.set_data(self.virtual_images[int(scan_slider.val)-1] * mask)
                ax.set_title(f'Scan {int(scan_slider.val)}')
                fig.canvas.draw_idle()
            # Keep references alive so interactive widgets continue to work.
            self._virtual_img_fig = fig
            self._virtual_img_ax = ax
            self._virtual_img_display = img_display
            self._virtual_img_slider = scan_slider
            self._virtual_img_slider_cid = scan_slider.on_changed(update)
            ax.axis('off')
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
    
    def find_img_shift_virtual_imgs(self, imgs: list[np.ndarray] | None = None, sub_region: tuple[int, int, int, int] | None = None, apply_hann_window: bool =False, print_shifts: bool =True, **kwargs) -> np.ndarray:
        '''Find the image shift for all scans using a virtual image signal
        imgs: sequence of 2D numpy array of virtual images. If None, will use self.virtual_images
        sub_region: tuple of (x0, x1, y0, y1) defining the sub-region to use for alignment. If None, will use the entire image.
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
        
        for i in tqdm(range(1, N_imgs), desc="Finding image shifts"):
            reference_img = imgs[i-1]
            moving_img = imgs[i]
            if sub_region is not None:
                x0, x1, y0, y1 = sub_region
                reference_img = reference_img[y0:y1, x0:x1]
                moving_img = moving_img[y0:y1, x0:x1]
            if apply_hann_window:
                data_shape = reference_img.shape
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


    

    def align_scan_positions(self, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, sub_region: tuple[int, int, int, int] | None = None,
                             apply_hann_window: bool =False, print_shifts: bool =True, print_cropping_info: bool = True, **kwargs) -> None:
        '''High-level function to align all scans in the ta4D dataset.
        **kwargs: Additional arguments passed to skimage.registration.phase_cross_correlation.'''
        max_workers = min(os.cpu_count() or 1, self.N_scans)
        self.get_virtual_images_parallel(max_workers=max_workers, type=type, center=center, radius=radius)
        self.find_img_shift_virtual_imgs(sub_region=sub_region, apply_hann_window=apply_hann_window, print_shifts=print_shifts, **kwargs)
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
                        type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, 
                        sub_region: tuple | None = None, apply_hann_window: bool =False,
                        refine_radius: int = 5, threshold: float = 1.5, fit_plane: bool = True, upsampling: int = 1,
                        use_numba: bool = True, **kwargs) -> np.ndarray:
        '''High-level function to align all diffraction patterns in all scans.
        First get the dp origins using get_dp_origins, then align using the origins.
        center: Tuple of (x, y) coordinates for virtual images and initial center guess. If None, will use the center of the image array.
        sub_region: tuple of (x0, x1, y0, y1) defining the sub-region to use for alignment. If None, will use the entire image.
        refine_radius: int, radius around the center to consider for CoM calculation
        threshold: float, threshold to generate binary mask for CoM calculation
        fit_plane: bool, whether to fit a plane to the origins before alignment
        upsampling: Factor by which to upsample the diffraction patterns before alignment
        use_numba: bool, whether to use numba for acceleration
        **kwargs: Additional arguments to pass to phase_cross_correlation in find_img_shift_virtual_imgs. Note: do not pass upsample_factor in kwargs.
        '''
        # First, align the scans in real space
        self.align_scan_positions(type=type, center=center, radius=radius, sub_region=sub_region, apply_hann_window=apply_hann_window, **kwargs)
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
    
    #========================== EWPC functions ==========================#
    
    @staticmethod
    def convert_dp_to_ewpc(dp: np.ndarray, useWindow: bool = True, minlog: float = 0.1, n_workers: int = -1):
        '''
        Applies cepstral transform to the data in parallel.

        :Parameters:
            dp : ndarray of shape (y, x, ky, kx)
                4D-STEM dataset.
            useWindow : bool
                Whether to apply a Hann window before FFT.
            minlog : float
                Small offset used in log transform to avoid log(0).
            n_workers: int
                Number of workers for the paralell computing. -1 to use all the CPUs.

        :Return:
            cep : ndarray
                Cepstral transformed 4D dataset.
        '''
        cep = np.zeros(dp.shape, dtype=np.float32)
        total_points = dp.shape[0] * dp.shape[1]
        scan_indices = list(np.ndindex(dp.shape[:2]))

        results = Parallel(n_jobs=n_workers)(
            delayed(ewpc2D)(dp[j, k], useWindow, minlog)
            for j, k in tqdm(scan_indices, total=total_points, desc="Converting to EWPC")
        )
        for (j, k), cep_jk in zip(scan_indices, results):
            cep[j, k] = cep_jk
        return cep

    def sum_dps_ewpc(self, interval=1, *args, **kwargs) -> np.ndarray:
        '''Convert all diffraction patterns in all scans to EWPC and sum them. Useful for strain mapping with EWPC.
        interval: int, the summing interval. If >1, will sum every 'interval' scans. For debug.
        *args, **kwargs: Additional parameters to pass to convert_dp_to_ewpc for the EWPC transform.
        Returns:
        sum_ewpc: 4D numpy array of the summed EWPC-transformed data.
        '''
        assert hasattr(self, 'datasets_cropped'), "datasets_cropped not found. Please run align_scan_positions() first."
        self.sum_ewpc = np.zeros(self.datasets_cropped[0].shape, dtype=self.datasets_cropped[0].dtype)
        for i in tqdm(range(0, self.N_scans, interval), desc="Computing EWPC sum"):
            dataset = np.ascontiguousarray(self.datasets_cropped[i])
            cep = self.convert_dp_to_ewpc(dataset, *args, **kwargs)
            self.sum_ewpc += cep
        return self.sum_ewpc
