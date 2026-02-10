import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.filters import window
from scipy.ndimage import shift
from numba import njit, prange, objmode

class ta4D_datasets:
    def __init__(self, data: list[np.ndarray]):
        """data: List of 4D data array to be processed.
        """
        self.N_scans = len(data)
        self.datasets = data


    def get_virtual_images(self, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None) -> list[np.ndarray]:
        """Get virtual images from all ta4D datasets for real-space alignment.
        type: Type of virtual image to generate. Options are 'brightfield' or 'darkfield'
        center: Center of the virtual image aperture. If None, will use the center of the dataset
        radius: Radius of the virtual image aperture. For 'brightfield', this is the radius of the brightfield disk. 
        For 'darkfield', this is the inner radius of the darkfield annulus. If None, for 'brightfield', will use 2 px; 
        for 'darkfield', will integrate from 1/2 of the maximum radius to the maximum radius.
        Returns:
        virtual_images: List of 2D numpy arrays of virtual images.
        """
        if center is None:
            center = (self.datasets[0].shape[2]//2, self.datasets[0].shape[3]//2)
        # Generate virtual mask according to type (diffraction-plane shape)
        virtual_mask = np.zeros((self.datasets[0].shape[2], self.datasets[0].shape[3]), dtype=bool)
        Y, X = np.ogrid[:self.datasets[0].shape[2], :self.datasets[0].shape[3]]
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
        
        # Generate virtual images
        self.virtual_images = []
        for i, dataset in enumerate(self.datasets):
            print(f'Generating virtual image for scan {i+1}/{self.N_scans}...')
            # dataset shape expected (Ny, Nx, Qy, Qx); virtual_mask (Qy, Qx) broadcasts.
            virtual_img = np.sum(dataset * virtual_mask, axis=(2, 3))
            self.virtual_images.append(virtual_img)

        return self.virtual_images
    
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
        shifts = np.zeros((N_imgs, 2), dtype=np.int16)
        
        for i in range(1, N_imgs):
            reference_img = imgs[i-1]
            moving_img = imgs[i]
            if apply_hann_window:
                hann_window = window('hann', data_shape)
                reference_img = reference_img * hann_window
                moving_img = moving_img * hann_window
            shift_i, _, _ = phase_cross_correlation(reference_img, moving_img, **kwargs)
            shifts[i] = shift_i + shifts[i-1]  # Accumulate shifts
            if print_shifts:
                print(f'Shift of scan {i+1}: (y, x) = ({shifts[i,0]}, {shifts[i,1]})')
    
        return shifts
    
    def align_scans_with_shifts(self, shifts: np.ndarray, print_cropping_info: bool = True) -> None:
        '''Align the scans using the provided shifts
        shifts: numpy array of shape (N_scans, 2), the shifts for each scan, must be integers
        print_cropping_info: bool, whether to print the cropping information for each scan
        '''       
        y_shifts = shifts[:, 0]
        x_shifts = shifts[:, 1]
        data_shape = self.datasets[0].shape[:2]  
        # Determine the cropping region
        y_min = max(0, max(y_shifts))
        y_max = min(data_shape[0], data_shape[0] + min(y_shifts))
        x_min = max(0, max(x_shifts))
        x_max = min(data_shape[1], data_shape[1] + min(x_shifts))
        # Apply the shifts and cropping to all datasets
        for i in range(self.N_scans):
            y_shift = y_shifts[i]
            x_shift = x_shifts[i]
            y0 = y_min - y_shift
            y1 = y_max - y_shift
            x0 = x_min - x_shift
            x1 = x_max - x_shift
            self.datasets[i] = self.datasets[i][y0:y1, x0:x1, :, :]
            if print_cropping_info:
                print(f'Scan {i+1} cropped from original indices of {y0}:{y1}, {x0}:{x1}, yielding a shape: {self.datasets[i].shape}')

    def align_scan_positions(self, type: str = 'brightfield', center: tuple | None = None, radius: int | None = None, 
                             apply_hann_window: bool =False, print_shifts: bool =True, print_cropping_info: bool = True, **kwargs) -> None:
        '''High-level function to align all scans in the ta4D dataset.
        **kwargs: Additional arguments passed to skimage.registration.phase_cross_correlation.'''
        self.get_virtual_images(type=type, center=center, radius=radius)
        shifts = self.find_img_shift_virtual_imgs(apply_hann_window=apply_hann_window, print_shifts=print_shifts, **kwargs)
        self.align_scans_with_shifts(shifts, print_cropping_info=print_cropping_info)


    def align_scans_dps(self, method: str = 'CoM', mask: tuple | None = None, 
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
