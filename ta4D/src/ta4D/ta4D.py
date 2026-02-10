from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *
from autoscript_tem_toolkit.context_managers import blanker_context
import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

class ta4D:
    """Tilt-averaged 4D-STEM data acquisition and processing class."""
    def __init__(self, microscope, calibration=None, metadata=None):
        """microscope: AutoScript TemMicroscopeClient object for microscope control
           calibration: Path to calibration file pkl for beam tilt and diffraction shift corrections
           metadata: Dictionary containing experiment metadata"""
        self.microscope = microscope
        if calibration is not None:
            with open(calibration, 'rb') as f:
                self.calibration = pickle.load(f)
        else:
            self.calibration = {}
        self.metadata = metadata if metadata is not None else {}

    def acquire_img(self, save_file: str | None, camera_detector: str = CameraType.FLUCAM, size: int = 1024, exposure_time: float = 0.1, plot: bool = True, title: str | None = None, **kwargs) -> AdornedImage:
        '''Acquire an image with autoscript with selected camera type
        Parameters
        ----------
        save_file : str | None
            File path to save the acquired image. If None, image is not saved.
        camera_detector : str
            Camera type to use for acquisition. Default is FLUCAM. Acceptable type: CameraType.FLUCAM, CameraType.BM_CETA, CameraType.BM_Empad, CameraType.EF_CCD
        size : int
            Size of the acquired image. Default is 1024.
        exposure_time : float
            Exposure time for the camera in seconds. Default is 0.1.
        plot : bool
            Whether to plot the acquired image. Default is True.
        **kwargs:
            Additional keyword arguments passed into matplotlib imshow function.
        Returns
        -------
        AdornedImage
            The acquired image. 
        '''
        try: 
            with blanker_context(self.microscope.optics.blanker):
                img = self.microscope.acquisition.acquire_camera_image(camera_detector=camera_detector, size=size, exposure_time=exposure_time)
        except Exception as e:
            print(f'Acquisition failed. {e}')
            return
        # Save
        if save_file is not None:
            img.save(save_file)

        # Plot
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(img.data, **kwargs)
            ax.set_axis_off()
            if title is not None:
                ax.set_title(title)
            plt.show()

        return img

    def calibrate_df_shift(self, shift_strength: float, save_file: str | None, *args, save_calibration: bool | str = False, **kwargs):
        original_df_shift = self.microscope.optics.deflectors.image_tilt
        ref_file = f'{save_file}_shift_ref.tiff' if save_file is not None else None
        # Take reference
        original_ref = self.acquire_img(ref_file, *args, title='Reference', **kwargs)
        # Make shift list, 0, 1 are +-x, 2, 3 are +-y
        shift_list = [original_df_shift + (shift_strength, 0), 
                    original_df_shift + (-shift_strength, 0), 
                    original_df_shift + (0, shift_strength), 
                    original_df_shift + (0, -shift_strength)]
        shift_factor_offset = []
        for shift in shift_list:
            shift_file =  f'{save_file}_shift_{shift}.tiff' if save_file is not None else None
            self.microscope.optics.deflectors.image_tilt = shift
            title = f'Diffraction shift: {shift}'
            img = self.acquire_img(shift_file, *args, title=title, **kwargs)
            shift, _, _ = phase_cross_correlation(original_ref.data, img.data, normalization=None)
            y, x = -shift
            shift_factor = (x ** 2 + y ** 2) ** 0.5 / abs(shift_strength)
            offset = math.atan2(y, x) # In rad
            shift_factor_offset.append((shift_factor, offset))
        self.microscope.optics.deflectors.image_tilt = original_df_shift

        x_factor = (shift_factor_offset[0][0] + shift_factor_offset[1][0]) / 2
        x_offset = (shift_factor_offset[0][1] + shift_factor_offset[1][1] - np.pi) / 2
        y_factor = (shift_factor_offset[2][0] + shift_factor_offset[3][0]) / 2
        y_offset = (shift_factor_offset[2][1] + shift_factor_offset[3][1] - np.pi) / 2

        # Update calibration dictionary
        HT = self.metadata.get('HT', 'default')
        CL = self.metadata.get('Camera length', 'default')
        self.calibration[HT][CL]['Diffraction shift'] = (x_factor, x_offset), (y_factor, y_offset)

        # Save calibration
        if save_calibration is not False:
            save_path = save_calibration if isinstance(save_calibration, str) else 'ta4D_calibration.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(self.calibration, f)
            print(f'Calibration saved to {os.path.abspath(save_path)}')

        return (x_factor, x_offset), (y_factor, y_offset)
            


    def calibrate_beam_tilt(self, tilt_strength: float, save_file: str | None, save_calibration: bool | str = False, *args, **kwargs):
        original_beam_tilt = self.microscope.optics.deflectors.beam_tilt
        ref_file = f'{save_file}_tilt_ref.tiff' if save_file is not None else None
        original_ref = self.acquire_img(ref_file, *args, title='Reference', **kwargs)
        tilt_list = [original_beam_tilt + (tilt_strength, 0), 
                    original_beam_tilt + (-tilt_strength, 0), 
                    original_beam_tilt + (0, tilt_strength), 
                    original_beam_tilt + (0, -tilt_strength)]
        tilt_factor_offset = []
        for tilt in tilt_list:
            tilt_file = f'{save_file}_tilt_{tilt}.tiff' if save_file is not None else None
            self.microscope.optics.deflectors.beam_tilt = tilt
            title = f'Beam tilt: {tilt}'
            img = self.acquire_img(tilt_file, *args, title=title, **kwargs)
            shift, _, _ = phase_cross_correlation(original_ref.data, img.data, normalization=None)
            y, x = -shift
            tilt_factor = (x ** 2 + y ** 2) ** 0.5 / abs(tilt_strength)
            offset = math.atan2(y, x) # In rad
            tilt_factor_offset.append((tilt_factor, offset))
        self.microscope.optics.deflectors.beam_tilt = original_beam_tilt

        x_factor = (tilt_factor_offset[0][0] + tilt_factor_offset[1][0]) / 2
        x_offset = (tilt_factor_offset[0][1] + tilt_factor_offset[1][1] - np.pi) / 2
        y_factor = (tilt_factor_offset[2][0] + tilt_factor_offset[3][0]) / 2
        y_offset = (tilt_factor_offset[2][1] + tilt_factor_offset[3][1] - np.pi) / 2

        # Update calibration dictionary
        HT = self.metadata.get('HT', 'default')
        CL = self.metadata.get('Camera length', 'default')
        self.calibration[HT][CL]['Beam tilt'] = (x_factor, x_offset), (y_factor, y_offset)

        # Save calibration
        if save_calibration is not False:
            save_path = save_calibration if isinstance(save_calibration, str) else 'ta4D_calibration.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(self.calibration, f)
            print(f'Calibration saved to {os.path.abspath(save_path)}')

        return (x_factor, x_offset), (y_factor, y_offset)

    def correct_beam_tilt_with_shift(self, tilt_strength: float):
        # Calculate shifts given any tilt strength tiltx, tilty
        # The beam tilt and diffraction shift calibrations need to be done beforehand
        HT = self.metadata.get('HT', 'default')
        CL = self.metadata.get('Camera length', 'default')
        calibration = self.calibration.get(HT, {}).get(CL, {})
        if 'Beam tilt' not in calibration or 'Diffraction shift' not in calibration:
            raise ValueError('Beam tilt and diffraction shift must be calibrated beforehand!')
        tilt_factor_offset = calibration['Beam tilt']
        shift_factor_offset = calibration['Diffraction shift']
        tiltx, tilty = tilt_strength
        x_tilt, y_tilt = tiltx * tilt_factor_offset[0][0], -tilty * tilt_factor_offset[1][0]
        # Rotate back to image coordinate, use x angle only since they are close enough
        tilt_offset = tilt_factor_offset[0][1]
        x = x_tilt * math.cos(tilt_offset) - y_tilt * math.sin(tilt_offset)
        y = x_tilt * math.sin(tilt_offset) + y_tilt * math.cos(tilt_offset)
        # Calculate beam shift needed
        shift_offset = -shift_factor_offset[0][1]
        x_shift = x * math.cos(shift_offset) - y * math.sin(shift_offset)
        y_shift = x * math.sin(shift_offset) + y * math.cos(shift_offset)
        shiftx, shifty = x_shift / shift_factor_offset[0][0], y_shift / shift_factor_offset[1][0]

        return shiftx, shifty

    def tilt_azimuth(self, tilt_strength: float, n_tilt: int, correct_shift: bool = True, acquire_flucam: bool = False, *args, **kwargs):
        azimuth_angle = np.arange(0, 2*np.pi, 2*np.pi/n_tilt)
        tilt_x = np.cos(azimuth_angle) * tilt_strength
        tilt_y = np.sin(azimuth_angle) * tilt_strength
        # Store the original beam tilt and df shift
        original_beam_tilt = self.microscope.optics.deflectors.beam_tilt
        original_df_shift = self.microscope.optics.deflectors.image_tilt
        # Tilt with df shift for acquisition
        for i in range(n_tilt):
            tilt = tilt_x[i], tilt_y[i]
            self.microscope.optics.deflectors.beam_tilt = original_beam_tilt + tilt
            if correct_shift:
                shift = self.correct_beam_tilt_with_shift(tilt)
                self.microscope.optics.deflectors.image_tilt = original_df_shift + shift
            
            if acquire_flucam:
                save_file = kwargs.pop('save_file', None)
                flucam_file = f'{save_file}_tilt_{i+1}_flucam.tiff' if save_file is not None else None
                self.acquire_img(flucam_file, *args, title=f'ta4D tilt {i+1}', **kwargs)

            print(f'TEM is ready to take ta4D scan: {i+1} of {n_tilt}')
            input('Press any key to proceed to the next scan when finished...')

        # Restore the original tilt and shift
        self.microscope.optics.deflectors.beam_tilt = original_beam_tilt 
        self.microscope.optics.deflectors.image_tilt = original_df_shift 
        print('ta4D scan completed!')
        


    def acquire_ta4D(self, tilt_ang: float, n_tilt: int, correct_shift: bool = True, acquire_flucam: bool = False, *args, **kwargs):
        '''High-level function to acquire ta4D datasets. 
        tilt_ang: Targeted tilt in mrad or px. If angular_calibration is provided in metadata, will tilt to the angle based on that.
        n_tilt: Number of tilt angles to acquire.
        correct_shift: Whether to apply diffraction shift correction during acquisition. Usually this should be True. Turn it off for debugging.
        acquire_flucam: Whether to acquire Flucam images during the tilt series. Turn it on for debugging.'''
        angle_cal = self.metadata.get('angular_calibration', 1.0) # mrad/px
        px_to_tilt = tilt_ang / angle_cal
        HT = self.metadata.get('HT', 'default')
        CL = self.metadata.get('Camera length', 'default')
        calibration = self.calibration.get(HT, {}).get(CL, {})
        if 'Beam tilt' not in calibration or 'Diffraction shift' not in calibration:
            # Run calibration with default strength
            print('Beam tilt and/or diffraction shift calibration not found for current HT and Camera length. Running calibration with default strengths.')
            print('Calibrating diffraction shift with 0.03...')
            self.calibrate_df_shift(0.03, None)
            print('Calibrating beam tilt with 0.03...')
            self.calibrate_beam_tilt(0.03, None)
            calibration = self.calibration.get(HT, {}).get(CL, {})

        tilt_factor_offset = calibration['Beam tilt']
        shift_factor_offset = calibration['Diffraction shift']

        tilt_strength = px_to_tilt / tilt_factor_offset[0][0] 
        # Take scans
        print(f'Acquiring ta4D with {tilt_ang} mrad of beam tilt, {n_tilt} scans.')
        with blanker_context(self.microscope.optics.blanker):
            self.tilt_azimuth(tilt_strength, n_tilt, correct_shift, acquire_flucam, *args, **kwargs)


    