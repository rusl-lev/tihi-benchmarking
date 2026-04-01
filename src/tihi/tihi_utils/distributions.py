from scipy.optimize import least_squares
from scipy.special import wofz
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class GaussianFitter():
    def __init__(self, full_x_vals, x_vals, y_vals, params, bounds, residual='default', max_iter=100, verbose=False):
        """
        Gaussian peak fitting class.
        
        :param InterpolatedData (object) : Interpolated data object containing x_val and y_val.
        :param peaks (array_like) : Indices of peaks in the data.
        :param max_iter (int, optional (default=100)) : Maximum number of iterations for fitting.
        :attribute x_vals (ndarray) : X values from InterpolatedData.
        :attribute y_vals (ndarray) : Y values from InterpolatedData.
        :attribute centers (ndarray) : Initial centers of Gaussian peaks.
        :attribute amplitudes (ndarray) : Initial amplitudes of Gaussian peaks.
        :attribute sigmas (ndarray) : Initial standard deviations of Gaussian peaks.
        :attribute params (ndarray) : Array of initial parameters for least squares fitting.
        :attribute start_params (list) : Flattened list of initial parameters.
        :attribute decompositions (list) : List to store individual Gaussian functions.
        :attribute result (None) : Placeholder for fitting result.
        """
        self.full_x_vals = full_x_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
                
        self.approximator(max_iter, residual)
        
    def approximator(self, max_iter, residual):
        """
        Perform Gaussian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        if residual == 'default':
            self.params = least_squares(self.residual,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error
        elif residual == 'log':
            self.params = least_squares(self.residual_log,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual_log(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error

        print(self.params)
        self.results = np.array([self.gaussian_sum(x, self.params) for x in self.full_x_vals])
        
        return error
    
    def gaussian(self, x, center, amplitude, gauss_width):
        """
        Calculate a Gaussian function.
        
        :param x (ndarray) : X values.
        :param center (float) : Center of the Gaussian function.
        :param amplitude (float) : Amplitude of the Gaussian function.
        :param sigma (float) : Standard deviation of the Gaussian function.
        :return (ndarray) : Calculated Gaussian function values.
        """
        # amplitude = amplitude * (-1.0)
        sigma = gauss_width / np.sqrt(2 * np.log(2))
        return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

    def gaussian_sum(self, x, params):
        """
        Calculate the sum of Gaussian functions.
        
        :param x (ndarray) : X values.
        :param params (ndarray) : Array of parameters for Gaussian functions.
        :return (ndarray) : Sum of Gaussian functions.
        """
        params = params.flatten().tolist()
        params = [params[i:i + 3] for i in range(0, len(params), 3)]
        self.decompositions = [self.gaussian(x, center, amp, sigma) for center, amp, sigma in params]
        return np.sum(self.decompositions, axis=0)

    def residual(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Gaussian fit.
        
        :param params (ndarray) : Array of parameters for Gaussian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return y_vals - self.gaussian_sum(x_vals, params)

    def residual_log(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Gaussian fit.
        
        :param params (ndarray) : Array of parameters for Gaussian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return np.log10(y_vals) - np.log10(self.gaussian_sum(x_vals, params))


class LorentzianFitter():
    """
    Lorentzian peak fitting class.
    
    :param InterpolatedData (object) : Interpolated data object containing x_val and y_val.
    :param peaks (array_like) : Indices of peaks in the data.
    :param max_iter (int, optional (default=100)) : Maximum number of iterations for fitting.
    :attribute x_vals (ndarray) : X values from InterpolatedData.
    :attribute y_vals (ndarray) : Y values from InterpolatedData.
    :attribute centers (ndarray) : Initial centers of Lorentzian peaks.
    :attribute amplitudes (ndarray) : Initial amplitudes of Lorentzian peaks.
    :attribute gammas (list) : Initial full width at half maximum (FWHM) of Lorentzian peaks.
    :attribute params (ndarray) : Array of initial parameters for least squares fitting.
    :attribute start_params (list) : Flattened list of initial parameters.
    :attribute decompositions (list) : List to store individual Lorentzian functions.
    """
    def __init__(self, full_x_vals, x_vals, y_vals, params, bounds, residual='default', max_iter=100, verbose=False):

        self.full_x_vals = full_x_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
                
        self.approximator(max_iter, residual)
        
    def approximator(self, max_iter, start_params, bounds, x_vals, y_vals):
        """
        Perform Lorentzian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        if residual == 'default':
            self.params = least_squares(self.residual,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error
        elif residual == 'log':
            self.params = least_squares(self.residual_log,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual_log(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error

        print(self.params)
        self.results = np.array([self.lorentzian_sum(x, self.params) for x in self.full_x_vals])
        
        return error
    
    def lorentzian(self, x, center, amplitude, lorentz_width):
        """
        Calculate a Lorentzian function.
        
        :param x (ndarray) : X values.
        :param center (float) : Center of the Lorentzian function.
        :param amplitude (float) : Amplitude of the Lorentzian function.
        :param gamma (float) : Full width at half maximum (FWHM) of the Lorentzian function.
        :return (ndarray) : Calculated Lorentzian function values.
        """
        # amplitude = amplitude * (-1.0)
        gamma = lorentz_width / 2
        # return amplitude * (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)
        return (amplitude * gamma ** 2) / (gamma ** 2 + (x - center) ** 2)

    def lorentzian_sum(self, x, params):
        """
        Calculate the sum of Lorentzian functions.
        
        :param x (ndarray) : X values.
        :param params (ndarray) : Array of parameters for Lorentzian functions.
        :return (ndarray) : Sum of Lorentzian functions.
        """
        params = params.tolist()
        params = [params[i:i + 3] for i in range(0, len(params), 3)]
        decompositions = [self.lorentzian(x, centre, amp, gamma) for centre, amp, gamma in params]
        return np.sum(decompositions, axis=0)

    def residual(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Lorentzian fit.
        
        :param params (ndarray) : Array of parameters for Lorentzian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return y_vals - self.lorentzian_sum(x_vals, params)
    
    def residual_log(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Lorentzian fit.
        
        :param params (ndarray) : Array of parameters for Lorentzian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return np.log10(y_vals) - np.log10(self.lorentzian_sum(x_vals, params))
    
class VoigtFitter():
    def __init__(self, full_x_vals, x_vals, y_vals, params, bounds, residual='default', max_iter=100, verbose=False):
        """
        Voigt peak fitting class.
        
        :param InterpolatedData (object) : Interpolated data object containing x_val and y_val.
        :param peaks (array_like) : Indices of peaks in the data.
        :param max_iter (int, optional (default=50)) : Maximum number of iterations for fitting.
        :attribute x_vals (ndarray) : X values from InterpolatedData.
        :attribute y_vals (ndarray) : Y values from InterpolatedData.
        :attribute centers (ndarray) : Initial centers of Voigt peaks.
        :attribute amplitudes (ndarray) : Initial amplitudes of Voigt peaks.
        :attribute gauss_widths (ndarray) : Initial Gaussian widths of Voigt peaks.
        :attribute lorentz_widths (ndarray) : Initial Lorentzian widths of Voigt peaks.
        :attribute params (ndarray) : Array of initial parameters for least squares fitting.
        :attribute start_params (list) : Flattened list of initial parameters.
        :attribute decompositions (list) : List to store individual Voigt functions.
        """
        self.full_x_vals = full_x_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
                
        self.approximator(max_iter, residual)
        
    def approximator(self, max_iter):
        """
        Perform Voigt fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        if residual == 'default':
            self.params = least_squares(self.residual,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error
        elif residual == 'log':
            self.params = least_squares(self.residual_log,
                                self.params, args=(self.x_vals, self.y_vals),
                                bounds=self.bounds,
                                ftol=1e-9, xtol=1e-9, loss='soft_l1',
                                f_scale=0.1, max_nfev=max_iter).x
            error = np.mean(np.abs(self.residual_log(self.params, self.x_vals, self.y_vals)))
            print("the error for this run is: ", error)
            self.error = error

        print(self.params)
        self.results = np.array([self.voigt_sum(x, self.params) for x in self.full_x_vals])
        
        return error
    
    def voigt(self, x, center, amplitude, gauss_width, lorentz_width):
        """
        Calculate a Voigt profile using Faddeeva function approximation.
        
        :param x (ndarray) : X values.
        :param center (float) : Center of the Voigt profile.
        :param amplitude (float) : Amplitude of the Voigt profile.
        :param gauss_width (float) : Gaussian component width of the Voigt profile.
        :param lorentz_width (float) : Lorentzian component width of the Voigt profile.
        :return (ndarray) : Calculated Voigt profile values.
        """
        sigma = gauss_width / np.sqrt(2 * np.log(2))
        gamma = lorentz_width / 2
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2) + 1e-20)
        return amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi) + 1e-20)

    def voigt_sum(self, x, params):
        """
        Calculate the sum of Voigt profiles.
        
        :param x (ndarray) : X values.
        :param params (ndarray) : Array of parameters for Voigt profiles.
        :return (ndarray) : Sum of Voigt profiles.
        """
        params = params.tolist()
        params = [params[i:i + 4] for i in range(0, len(params), 4)]
        self.decompositions = [self.voigt(x, centre, amp, gw, lw) for centre, amp, gw, lw in params]
        return np.sum(self.decompositions, axis=0)

    def residual(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Voigt fit.
        
        :param params (ndarray) : Array of parameters for Voigt profiles.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return y_vals - self.voigt_sum(x_vals, params)

    def residual_log(self, params, x_vals, y_vals):
        """
        Calculate residual between data and Lorentzian fit.
        
        :param params (ndarray) : Array of parameters for Lorentzian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        return np.log10(y_vals) - np.log10(self.lorentzian_sum(x_vals, params))


approximators_dict = {
    'gauss': GaussianFitter,
    'lorentz': LorentzianFitter,
    'voigt': VoigtFitter
}


def complex_fitting(
    data: np.ndarray, 
    peaks: np.ndarray, 
    spec_bounds: np.ndarray, 
    peak_rtol: Optional[float] = 5e-02, 
    max_iter: Optional[int] = 100,
    residual: Optional[str] = 'default',
    verbose: bool = False
) -> Tuple[np.ndarray, list, float]:
    
    x_vals = data[:,0]
    y_vals = data[:,1]
    final_approximation = np.array(x_vals, np.zeros_like(y_vals)).T
                    
    # initial parameters
    centers = peaks[:,0]
    amplitudes = peaks[:,1]
    lorentz_widths = np.random.rand(*amplitudes.shape)
    gauss_widths = np.random.rand(*amplitudes.shape)
    approximation_results = np.empty([spec_bounds.shape[0]-1, self.x_vals.shape[0]])
    output_parameters = []
    
    if verbose:
        print(f'Shape of the bounds array: ', self.bounds.shape)
        print(f'Bounds array: ', self.bounds)
        print(f'X values: {self.x_vals}')
        print()
        fig, axs = plt.subplots(self.bounds.shape[0], 1, figsize=(10, 20))

    for i in range(spec_bounds.shape[0]-1):
        x_ub = spec_bounds[i+1]
        x_lb = spec_bounds[i]
        allowed_dev = (x_ub - x_lb) * peak_rtol
        peak_mask = (centers >= x_lb) & (centers <= x_ub)
        centers_i = centers[peak_mask]
        amplitudes_i = amplitudes[peak_mask]
        lorentz_widths_i = lorentz_widths[peak_mask]
        gauss_widths_i = gauss_widths[peak_mask]
        # peak_deviation_bound = ([(peaks - allowed_dev), 0, 0], [(peaks + allowed_dev), np.inf, np.inf])
        mask = (x_vals >= x_lb) & (x_vals <= x_ub)
        x_masked = x_vals[mask]
        y_masked = y_vals[mask]
        min_error = 1e10
        bound_approximator = None
        approximation_i = None
        params_i = None

        parameters_dict = {
            'gauss': np.array([centers_i, amplitudes_i, gauss_widths_i]).T,
            'lorentz': np.array([centers_i, amplitudes_i, lorentz_widths_i]).T,
            'voigt': np.array([centers_i, amplitudes_i, gauss_widths_i, lorentz_widths_i]).T
        }

        bounds_dict = {
            'gauss': ([(peaks - allowed_dev), 0, 0], [(peaks + allowed_dev), np.inf, np.inf]),
            'lorentz': ([(peaks - allowed_dev), 0, 0], [(peaks + allowed_dev), np.inf, np.inf]),
            'voigt': ([(peaks - allowed_dev), 0, 0, 0], [(peaks + allowed_dev), np.inf, np.inf, np.inf])
        }
        
        for approximator in enumerate(approximators_dict):
            params = parameters_dict[approximator]
            bounds = bounds_dict[approximator]
            aprx = approximators_dict[approximator](x_vals, x_masked, y_masked, params, bounds, residual=residual, max_iter=max_iter, verbose=verbose)
            if min_error > aprx.error:
                min_error = aprx.error
                bound_approximator = aprx

        if verbose:
            print(f'Minimum error for bound ({x_lb}, {x_ub}) of {min_error} is produced by {bound_approximator}.')
        
        approximation_i = bound_approximator.results
        final_approximation[:,1] += approximation_i
        params_i = bound_approximator.params
        output_parameters.append(params_i)
        
        if verbose:
            print(f'Lower bound: {x_lb}; upper bound: {x_ub}')
            print(f'X masked: {x_masked}')
            print(f'Y masked: {y_masked}')
            print(f'Minimum error of {min_error} for the bound is produced by {bound_approximator}.')
            print(f'Paramerers for the bound: ', params_i)
            print()
            axs[i].plot(x_masked, y_masked, label="Spectrum")
            axs[i].plot(x_masked, approximation_i[mask], label="Fit")
            axs[i].plot(centers_i, amplitudes_i, color='k', marker='x', label="Initial Peaks")
            axs[i].plot(params_i[:,0], params_i[:,1], color='r', marker='x', label="Fitted Peaks")
            axs[i].set_ylabel('Signal amplitude')
    if verbose:
        axs[-1].plot(x_vals, y_vals, label="Spectrum")
        axs[-1].plot(final_approximation[:,0], final_approximation[:,1], label="Total Fit")
        axs[-1].set_xlabel('Wavenumbers [$cm^{-1}$]')
        plt.tight_layout()
        plt.legend()
        plt.show()

    rmsd = np.sqrt(np.sum((final_approximation[:,1] - y_vals) ** 2))

    return final_approximation, output_parameters, rmsd
