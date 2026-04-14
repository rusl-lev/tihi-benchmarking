from scipy.optimize import least_squares
from scipy.special import wofz, softmax
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

state_scope_map = {
    'init': 'bounded',
    'refit': 'full'
}

class GaussianFitter():
    def __init__(self, full_x_vals, full_y_vals, x_vals, y_vals, params, bounds, state, spec_rest=None, residual='default', max_iter=100, verbose=False):
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
        self.full_y_vals = full_y_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
        self.spec_rest = spec_rest
        self.residual_scope = state_scope_map[state]
        self.residual_type = residual
                
        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        """
        Perform Gaussian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        self.params = least_squares(self.residual,
                            self.params, args=(self.x_vals, self.y_vals, self.residual_scope, self.residual_type),
                            bounds=self.bounds,
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals, self.residual_scope, self.residual_type)))
        # print("The error for this run is: ", error)
        # self.error = error

        # print(self.params)
        # self.results = np.array([self.gaussian_sum(x, self.params) for x in self.full_x_vals])
        self.results = self.gaussian_sum(self.full_x_vals, self.params)
        
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
        decompositions = [self.gaussian(x, center, amp, sigma) for center, amp, sigma in params]
        return np.sum(decompositions, axis=0)

    def residual(self, params, x_vals, y_vals, scope, type):
        """
        Calculate residual between data and Gaussian fit.
        
        :param params (ndarray) : Array of parameters for Gaussian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        fit = self.gaussian_sum(x_vals, params) if self.residual_scope == 'bounded' else self.gaussian_sum(self.full_x_vals, params) + self.spec_rest
        reference = y_vals if self.residual_scope == 'bounded' else self.full_y_vals
        residual = reference - fit if self.residual_type == 'default' else np.log10(reference) - np.log10(fit)
        return residual


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
    def __init__(self, full_x_vals, full_y_vals, x_vals, y_vals, params, bounds, state, spec_rest=None, residual='default', max_iter=100, verbose=False):

        self.full_x_vals = full_x_vals
        self.full_y_vals = full_y_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
        self.spec_rest = spec_rest
        self.residual_scope = state_scope_map[state]
        self.residual_type = residual
                
        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        """
        Perform Lorentzian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        self.params = least_squares(self.residual,
                            self.params, args=(self.x_vals, self.y_vals, self.residual_scope, self.residual_type),
                            bounds=self.bounds,
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals, self.residual_scope, self.residual_type)))
        # print("The error for this run is: ", error)
        # self.error = error

        # print(self.params)
        # self.results = np.array([self.lorentzian_sum(x, self.params) for x in self.full_x_vals])
        self.results = self.lorentzian_sum(self.full_x_vals, self.params)
        
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

    def residual(self, params, x_vals, y_vals, scope, type):
        """
        Calculate residual between data and Lorentzian fit.
        
        :param params (ndarray) : Array of parameters for Lorentzian functions.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        fit = self.lorentzian_sum(x_vals, params) if self.residual_scope == 'bounded' else self.lorentzian_sum(self.full_x_vals, params) + self.spec_rest
        reference = y_vals if self.residual_scope == 'bounded' else self.full_y_vals
        residual = reference - fit if self.residual_type == 'default' else np.log10(reference) - np.log10(fit)
        return residual

    
class VoigtFitter():
    def __init__(self, full_x_vals, full_y_vals, x_vals, y_vals, params, bounds, state, spec_rest=None, residual='default', max_iter=100, verbose=False):
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
        self.full_y_vals = full_y_vals
        self.x_vals = x_vals
        self.y_vals = y_vals
                    
        # initial parameters
        self.params = params
        self.bounds = bounds
        self.output_params = []
        self.results = np.empty(self.full_x_vals.shape[0])
        self.error = 0
        self.spec_rest = spec_rest
        self.residual_scope = state_scope_map[state]
        self.residual_type = residual
                
        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        """
        Perform Voigt fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        self.params = least_squares(self.residual,
                            self.params, args=(self.x_vals, self.y_vals, self.residual_scope, self.residual_type),
                            bounds=self.bounds,
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        error = np.mean(np.abs(self.residual(self.params, self.x_vals, self.y_vals, self.residual_scope, self.residual_type)))
        # print("The error for this run is: ", error)
        # self.error = error

        # print(self.params)
        # self.results = np.array([self.voigt_sum(x, self.params) for x in self.full_x_vals])
        self.results = self.voigt_sum(self.full_x_vals, self.params)
        
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
        gamma = lorentz_width / 2.0
        
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2) + 1e-20)
        real_part = np.real(wofz(z))
        norm = sigma * np.sqrt(2 * np.pi)
        profile = amplitude * real_part / norm
        return profile

    def voigt_sum(self, x, params):
        """
        Calculate the sum of Voigt profiles.
        
        :param x (ndarray) : X values.
        :param params (ndarray) : Array of parameters for Voigt profiles.
        :return (ndarray) : Sum of Voigt profiles.
        """
        params = params.tolist()
        params = [params[i:i + 4] for i in range(0, len(params), 4)]
        decompositions = [self.voigt(x, centre, amp, gw, lw) for centre, amp, gw, lw in params]
        return np.sum(decompositions, axis=0)

    def residual(self, params, x_vals, y_vals, scope, type):
        """
        Calculate residual between data and Voigt fit.
        
        :param params (ndarray) : Array of parameters for Voigt profiles.
        :param x_vals (ndarray) : X values of the data.
        :param y_vals (ndarray) : Y values of the data.
        :return (ndarray) : Residual values.
        """
        fit = self.voigt_sum(x_vals, params) if self.residual_scope == 'bounded' else self.voigt_sum(self.full_x_vals, params) + self.spec_rest
        reference = y_vals if self.residual_scope == 'bounded' else self.full_y_vals
        residual = reference - fit if self.residual_type == 'default' else np.log10(reference) - np.log10(fit)
        return residual


# def complex_fitting(
#     data: np.ndarray, 
#     peaks: np.ndarray, 
#     spec_bounds: np.ndarray, 
#     peak_rtol: Optional[float] = 5e-02, 
#     max_iter: Optional[int] = 100,
#     residual: Optional[str] = 'default',
#     verbose: bool = False
# ) -> Tuple[np.ndarray, list, float]:
    
#     x_vals = data[:,0]
#     y_vals = data[:,1]
#     mask_outside = (x_vals >= 400) & (x_vals <= 3500)
#     x_vals = x_vals[mask_outside]
#     y_vals = y_vals[mask_outside]
#     final_approximation = np.array([x_vals, np.zeros_like(y_vals)]).T
                    
#     # initial parameters
#     centers = peaks[:,0]
#     amplitudes = peaks[:,1]
#     lorentz_widths = np.full_like(amplitudes, 15)
#     gauss_widths = np.full_like(amplitudes, 15)
#     output_parameters = []
    
#     if verbose:
#         print(f'Shape of the bounds array: ', spec_bounds.shape)
#         print(f'Bounds array: ', spec_bounds)
#         print(f'X values: {x_vals}')
#         print()

#     for i in range(spec_bounds.shape[0]-1):
#         x_ub = spec_bounds[i+1]
#         x_lb = spec_bounds[i]
#         allowed_dev = (x_ub - x_lb) * peak_rtol
#         peak_mask = (centers >= x_lb) & (centers < x_ub)
#         centers_i = centers[peak_mask]
#         n_peaks_i = centers_i.shape[0]
#         amplitudes_i = amplitudes[peak_mask]
#         lorentz_widths_i = lorentz_widths[peak_mask]
#         gauss_widths_i = gauss_widths[peak_mask]
#         # peak_deviation_bound = ([(peaks - allowed_dev), 0, 0], [(peaks + allowed_dev), np.inf, np.inf])
#         mask = (x_vals >= x_lb) & (x_vals <= x_ub)
#         x_masked = x_vals[mask]
#         y_masked = y_vals[mask]
#         min_error = 1e10
#         bound_approximator = None
#         approximation_i = None
#         params_i = None

#         parameters_dict = {
#             'gauss': np.array([centers_i, amplitudes_i, gauss_widths_i]).T.flatten().tolist(),
#             'lorentz': np.array([centers_i, amplitudes_i, lorentz_widths_i]).T.flatten().tolist(),
#             'voigt': np.array([centers_i, amplitudes_i, gauss_widths_i, lorentz_widths_i]).T.flatten().tolist()
#         }

#         centers_lb = centers_i - allowed_dev
#         centers_ub = centers_i + allowed_dev
#         amplitude_lower = np.full(n_peaks_i, 1e-10)
#         amp_lb = amplitudes_i * (1 - peak_rtol)
#         amp_ub = amplitudes_i * (1 + peak_rtol)
#         width_lower = np.full(n_peaks_i, 0.5)
#         infinity = np.full(n_peaks_i, np.inf)
#         bounds_dict = {
#             'gauss': (np.array([centers_lb, amplitude_lower, width_lower]).T.flatten().tolist(), np.array([centers_ub, infinity, infinity]).T.flatten().tolist()),
#             'lorentz': (np.array([centers_lb, amplitude_lower, width_lower]).T.flatten().tolist(), np.array([centers_ub, infinity, infinity]).T.flatten().tolist()),
#             'voigt': (np.array([centers_lb, amplitude_lower, width_lower, width_lower]).T.flatten().tolist(), np.array([centers_ub, infinity, infinity, infinity]).T.flatten().tolist())
#         }
        
#         for approximator in approximators_dict:
#             params = parameters_dict[approximator]
#             bounds = bounds_dict[approximator]
#             aprx = approximators_dict[approximator](x_vals, x_masked, y_masked, params, bounds, residual=residual, max_iter=max_iter, verbose=verbose)
#             if min_error > aprx.error:
#                 min_error = aprx.error
#                 bound_approximator = aprx

#         if verbose:
#             print(f'Minimum error for bound ({x_lb}, {x_ub}) of {min_error} is produced by {bound_approximator}.')
        
#         approximation_i = bound_approximator.results
#         final_approximation[:,1] += approximation_i
#         params_i = reshape_params(bound_approximator.params, bound_approximator)
#         output_parameters.append(params_i)
        
#         if verbose:
#             print(f'Lower bound: {x_lb}; upper bound: {x_ub}')
#             print(f'X masked: {x_masked}')
#             print(f'Y masked: {y_masked}')
#             print(f'Minimum error of {min_error} for the bound is produced by {bound_approximator}.')
#             print(f'Paramerers for the bound: ', params_i)
#             print()
#             fig = plt.figure(figsize=(10, 5))
#             plt.plot(x_masked, y_masked, label="Spectrum")
#             plt.plot(x_masked, approximation_i[mask], label="Fit")
#             plt.plot(centers_i, amplitudes_i, color='k', marker='x', label="Initial Peaks", linestyle='None')
#             plt.plot(params_i[:,0], params_i[:,1], color='r', marker='x', label="Fitted Peaks", linestyle='None')
#             plt.ylabel('Signal amplitude')
#             plt.xlabel('Wavenumbers [$cm^{-1}$]')
#             plt.title('Bound: ' + str(x_lb) + ' to ' + str(x_ub) + ' [$cm^{-1}$]')
#             plt.legend()
#             plt.show()
    
#     if verbose:
#         fig = plt.figure(figsize=(10,10))
#         plt.plot(x_vals, y_vals, label="Spectrum")
#         plt.plot(final_approximation[:,0], final_approximation[:,1], label="Total Fit")
#         plt.xlabel('Wavenumbers [$cm^{-1}$]')
#         plt.ylabel('Signal amplitude')
#         plt.tight_layout()
#         plt.legend()
#         plt.show()

#     rmsd = np.sqrt(np.mean(((final_approximation[:,1] - y_vals) / y_vals) ** 2))

#     return final_approximation, output_parameters, rmsd


# def sum_subpeak_fitting(state, approximators_array, x_vals, y_vals, full_x_vals, parameters_arg, bounds_arg, peak_excluded_spec, args_dict):
#     output_params = []
#     if state == 'init':
#         fits = np.zeros((len(approximators_array), full_x_vals.shape[0]))
#         for i, approx in enumerate(approximators_array):
#             params = parameters_arg[approx]
#             bounds = bounds_arg[approx]
#             approximator = approximators_dict[approx](full_x_vals, x_vals, y_vals, params, bounds, state, peak_excluded_spec, residual=args_dict['residual'], max_iter=args_dict['max_iter'], verbose=args_dict['verbose'])
#             fits[i] = approximator.results
#             output_params.append(reshape_params(approximator.params, approximator))
#         return fits, output_params
#     else:
#         sum_fits = np.zeros_like(full_x_vals)
#         for i, approx in enumerate(approximators_array):
#             params = parameters_arg[i]
#             bounds = bounds_arg[i]
#             approximator = approx(full_x_vals, x_vals, y_vals, params, bounds, state, peak_excluded_spec, residual=args_dict['residual'], max_iter=args_dict['max_iter'], verbose=args_dict['verbose'])
#             sum_fits += approximator.results
#             output_params.append(reshape_params(approximator.params, approximator))
#         return sum_fits, output_params


# def construct_bounds(approximators, params, allowed_dev):
#     output_bounds = []
#     for param in params:
#         num_params = len(params)
#         center_lb = params[0] - allowed_dev
#         center_ub = params[0] + allowed_dev
#         arr_lower = [center_lb, 1e-10] + [0.5] * (num_params - 2)
#         arr_upper = [center_ub] + [np.inf] * (num_params - 1)
#         output_bounds.append((arr_lower, arr_upper))
#     return output_bounds


# def residual_complex_fitting(params, x_vals, y_vals, spec_bounds, centers, amplitudes, lorentz_widths, gauss_widths, args_dict):
#     distributions = [list(approximators_dict.keys())[distr] for distr in params]
#     initial_approximation = np.zeros((centers.shape[0], x_vals.shape[0]))
#     final_approximation = np.array([x_vals, np.zeros_like(y_vals)]).T
#     peak_rtol = args_dict['peak_rtol']
#     initial_params = []
#     final_params = []
#     bounds_numbered = np.arange(spec_bounds.shape[0]-1, dtype=np.int_)
#     approximators_grouped = []
#     residual_type = args_dict['residual']
    
#     for i in bounds_numbered:
#         x_ub = spec_bounds[i+1]
#         x_lb = spec_bounds[i]
#         allowed_dev = (x_ub - x_lb) * peak_rtol
#         peak_mask = (centers >= x_lb) & (centers < x_ub)
#         mask = (x_vals >= x_lb) & (x_vals <= x_ub)
#         center_i = centers[peak_mask]
#         n_peaks_i = centers_i.shape[0]
#         amplitudes_i = amplitudes[peak_mask]
#         lorentz_widths_i = lorentz_widths[peak_mask]
#         gauss_widths_i = gauss_widths[peak_mask]
#         x_masked = x_vals[mask]
#         y_masked = y_vals[mask]
#         approximators = distributions[peak_mask]
#         approximators_grouped.append(approximators)
#         min_error = 1e10

#         parameters_dict = {
#             'gauss': np.array([centers_i, amplitudes_i, gauss_widths_i]).T.flatten().tolist(),
#             'lorentz': np.array([centers_i, amplitudes_i, lorentz_widths_i]).T.flatten().tolist(),
#             'voigt': np.array([centers_i, amplitudes_i, gauss_widths_i, lorentz_widths_i]).T.flatten().tolist()
#         }

#         centers_lb = centers_i - allowed_dev
#         centers_ub = centers_i + allowed_dev
#         amplitude_lower = np.full(n_peaks_i, 1e-10)
#         amp_lb = amplitudes_i * (1 - peak_rtol)
#         amp_ub = amplitudes_i * (1 + peak_rtol)
#         width_lower = np.full(n_peaks_i, 0.5)
#         infinity = np.full(n_peaks_i, np.inf)
#         gauss_lorentz_bounds = (np.array([centers_lb, amplitude_lower, width_lower]).T.flatten().tolist(), np.array([centers_ub, infinity, infinity]).T.flatten().tolist())
#         voigt_bounds = (np.array([centers_lb, amplitude_lower, width_lower, width_lower]).T.flatten().tolist(), np.array([centers_ub, infinity, infinity, infinity]).T.flatten().tolist())
        
#         bounds_dict = {
#             'gauss': gauss_lorentz_bounds,
#             'lorentz': gauss_lorentz_bounds,
#             'voigt': voigt_bounds
#         }

#         fits_i, params_i = sum_subpeak_fitting('init', approximators, x_masked, y_masked, x_vals, parameters_dict, bounds_dict, args_dict)
#         initial_approximation[peak_mask] += fits_i
#         initial_params.append(params_i)

#     rng = np.random.default_rng()
#     rng.shuffle(bounds_numbered)
#     for i in bounds_numbered:
#         parameters = initial_params[i]
#         approximators = approximators_grouped[i]
#         x_ub = spec_bounds[i+1]
#         x_lb = spec_bounds[i]
#         allowed_dev = (x_ub - x_lb) * peak_rtol
#         peak_mask = (centers >= x_lb) & (centers < x_ub)
#         mask = (x_vals >= x_lb) & (x_vals <= x_ub)
#         x_masked = x_vals[mask]
#         y_masked = y_vals[mask]
#         bounds = construct_bounds(approximators, parameters, allowed_dev)
#         fit_summation_mask = np.full(initial_approximation.shape[0], True)
#         fit_summation_mask[i] = False
#         fit_peak_excluded = np.sum(initial_approximation[fit_summation_mask], axis=0)

#         approximation_i, params_i = sum_subpeak_fitting('refit', approximators, x_masked, y_masked, x_vals, parameters, bounds, fit_peak_excluded, args_dict)
#         final_approximation[:,1] += approximation_i
#         final_params.append(params_i)

#     residual = y_vals - final_approximation if residual_type == 'default' else np.log10(y_vals) - np.log10(final_approximation)
    
#     return residual

# def complex_fitting_full(
#     data: np.ndarray, 
#     peaks: np.ndarray, 
#     spec_bounds: np.ndarray, 
#     peak_rtol: Optional[float] = 5e-02, 
#     max_iter: Optional[int] = 100,
#     residual: Optional[str] = 'default',
#     verbose: bool = False,
#     seed: int = 42
# ) -> Tuple[np.ndarray, list, float]:
    
#     x_vals = data[:,0]
#     y_vals = data[:,1]
#     mask_outside = (x_vals >= 400) & (x_vals <= 3500)
#     x_vals = x_vals[mask_outside]
#     y_vals = y_vals[mask_outside]
#     # final_approximation = np.array([x_vals, np.zeros_like(y_vals)]).T
                    
#     # initial parameters
#     centers = peaks[:,0]
#     amplitudes = peaks[:,1]
#     lorentz_widths = np.full_like(amplitudes, 15)
#     gauss_widths = np.full_like(amplitudes, 15)
#     rng = np.random.default_rng(seed=seed)
#     distr_sequence_init = rng.integers(0, 3, size=peaks.shape[0])
#     output_parameters = []
#     args_dict = {
#         'peak_rtol': peak_rtol, 
#         'max_iter': max_iter,
#         'residual': residual,
#         'verbose': verbose
#     }

    

#     distr_sequence_out = least_squares(residual_complex_fitting,
#                             distr_sequence_init, args=(x_vals, y_vals, spec_bounds, centers, amplitudes, lorentz_widths, gauss_widths, args_dict),
#                             bounds=(0, 2),
#                             ftol=1e-9, xtol=1e-9, loss='soft_l1',
#                             f_scale=0.1, max_nfev=max_iter).x

    
    
#     if verbose:
#         print(f'Shape of the bounds array: ', spec_bounds.shape)
#         print(f'Bounds array: ', spec_bounds)
#         print(f'X values: {x_vals}')
#         print()
    
    
#     if verbose:
#         fig = plt.figure(figsize=(10,10))
#         plt.plot(x_vals, y_vals, label="Spectrum")
#         plt.plot(final_approximation[:,0], final_approximation[:,1], label="Total Fit")
#         plt.xlabel('Wavenumbers [$cm^{-1}$]')
#         plt.ylabel('Signal amplitude')
#         plt.tight_layout()
#         plt.legend()
#         plt.show()

#     rmsd = np.sqrt(np.mean(((final_approximation[:,1] - y_vals) / y_vals) ** 2))

#     return final_approximation, output_parameters, rmsd


class ComplexFitterFull():
    
    def __init__(
        self, 
        data: np.ndarray, 
        peaks: np.ndarray, 
        spec_bounds: np.ndarray, 
        peak_rtol: Optional[float] = 5e-02, 
        max_iter: Optional[int] = 100,
        residual: Optional[str] = 'default',
        verbose: bool = False,
        seed: int = 42
    ):

        self.args_dict = {
            'peak_rtol': peak_rtol, 
            'max_iter': max_iter,
            'residual': residual,
            'verbose': verbose
        }
        
        self.approximators_dict = {
            'gauss': GaussianFitter,
            'lorentz': LorentzianFitter,
            'voigt': VoigtFitter
        }
        
        self.x_vals = data[:,0]
        self.y_vals = data[:,1]
        mask_outside = (self.x_vals >= 400) & (self.x_vals <= 3500)
        self.x_vals = self.x_vals[mask_outside]
        self.y_vals = self.y_vals[mask_outside]
        # final_approximation = np.array([x_vals, np.zeros_like(y_vals)]).T
                        
        # initial parameters
        self.centers = peaks[:,0]
        self.amplitudes = peaks[:,1]
        self.lorentz_widths = np.full_like(self.amplitudes, 15)
        self.gauss_widths = np.full_like(self.amplitudes, 15)
        rng = np.random.default_rng(seed=seed)
        self.distr_sequence_init = rng.integers(3, size=peaks.shape[0])
        self.spec_bounds = spec_bounds
        self.rmsd = 0
        self.iterations_distr = 0

        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        
        self.output_approx_params = least_squares(self.residual_complex_fitting,
                            self.distr_sequence_init, bounds=(0, 2),
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        
        return None

    def infer_params_from_approx(self, approximator):
        if isinstance(approximator, GaussianFitter) or isinstance(approximator, LorentzianFitter):
            num_params = 3
        else:
            num_params = 4
        return num_params
    
    def reshape_params(self, params, approximator):
        num_params = self.infer_params_from_approx(approximator)
        params = params.tolist()
        params = [params[i:i + num_params] for i in range(0, len(params), num_params)]
        return np.array(params)
    
    def construct_bounds(self, params, allowed_dev):
        output_bounds = []
        for param in params:
            num_params = len(param)
            center_lb = param[0] - allowed_dev
            center_ub = param[0] + allowed_dev
            arr_lower = [center_lb, 1e-10] + [0.5] * (num_params - 2)
            arr_upper = [center_ub] + [np.inf] * (num_params - 1)
            output_bounds.append((arr_lower, arr_upper))
        return output_bounds

    def sum_subpeak_fitting(self, state, approximators_array, x_masked, y_masked, parameters_arg, bounds_arg, peak_excluded_spec):
        output_params = []
        fits = np.zeros((len(approximators_array), self.x_vals.shape[0])) if state == 'init' else np.zeros_like(self.x_vals)
        fitted_approximators = []
        for i, approx in enumerate(approximators_array):
            params = parameters_arg[i]
            bounds = bounds_arg[i]
            
            approximator = self.approximators_dict[approx](
                self.x_vals, self.y_vals, x_masked, y_masked, 
                params, bounds, state, peak_excluded_spec, 
                residual=self.args_dict['residual'], max_iter=self.args_dict['max_iter'], verbose=self.args_dict['verbose']
            )
            output_params.append(approximator.params)

            if state == 'init':
                fits[i] = approximator.results
            elif state == 'refit':
                fits += approximator.results
            
            fitted_approximators.append(approximator)

        return fits, output_params, fitted_approximators

    def fit(self, state, i):
        x_ub = self.spec_bounds[i+1]
        x_lb = self.spec_bounds[i]
        peak_rtol = self.args_dict['peak_rtol']
        allowed_dev = (x_ub - x_lb) * peak_rtol
        peak_mask = (self.centers >= x_lb) & (self.centers < x_ub)
        mask = (self.x_vals >= x_lb) & (self.x_vals <= x_ub)
        x_masked = self.x_vals[mask]
        y_masked = self.y_vals[mask]
        approximators = self.distributions[peak_mask]
        fit_peak_excluded = 0

        # Construcing parameters depending on the fit state
        if state == 'init':
            centers_i = self.centers[peak_mask].tolist()
            amplitudes_i = self.amplitudes[peak_mask].tolist()
            lorentz_widths_i = self.lorentz_widths[peak_mask].tolist()
            gauss_widths_i = self.gauss_widths[peak_mask].tolist()

            parameters = [[center, amplitude] for center, amplitude in zip(centers_i, amplitudes_i)]
            for i, approximator in enumerate(approximators):
                if approximator == 'gauss':
                    parameters[i] += [gauss_widths_i[i]]
                elif approximator == 'lorentz':
                    parameters[i] += [lorentz_widths_i[i]]
                else:
                    parameters[i] += [gauss_widths_i[i], lorentz_widths_i[i]]
        
        elif state == 'refit':
            parameters = self.initial_params[i]
            fit_summation_mask = np.full(self.initial_approximation.shape[0], True)
            fit_summation_mask[i] = False
            fit_peak_excluded = np.sum(self.initial_approximation[fit_summation_mask], axis=0)

        bounds = self.construct_bounds(parameters, allowed_dev)
        fits_i, params_i, fitted_approximators = self.sum_subpeak_fitting(state, approximators, x_masked, y_masked, parameters, bounds, fit_peak_excluded)
        self.output_fitted_approximators[i] = fitted_approximators
        
        return fits_i, params_i, peak_mask

    def residual_complex_fitting(self, params):
        self.iterations_distr += 1
        self.distributions = np.array([list(self.approximators_dict.keys())[int(round(distr))] for distr in params])
        self.initial_approximation = np.zeros((self.centers.shape[0], self.x_vals.shape[0]))
        self.final_approximation = np.array([self.x_vals, np.zeros_like(self.y_vals)]).T
        self.initial_params = []
        self.final_params = []
        bounds_numbered = np.arange(self.spec_bounds.shape[0]-1, dtype=np.int_)
        residual_type = self.args_dict['residual']
        self.output_fitted_approximators = np.empty_like(bounds_numbered, dtype=object)
        
        for i in bounds_numbered:
            fits_i, params_i, peak_mask = self.fit('init', i)
            self.initial_approximation[peak_mask] += fits_i
            self.initial_params.append(params_i)
    
        rng = np.random.default_rng()
        rng.shuffle(bounds_numbered)
        for i in bounds_numbered:
            approximation_i, params_i, _ = self.fit('refit', i)
            self.final_approximation[:,1] += approximation_i
            self.final_params.append(params_i)
    
        residual = self.y_vals - self.final_approximation[:,1] if residual_type == 'default' else np.log10(self.y_vals) - np.log10(self.final_approximation[:,1])
        self.rmsd = np.sqrt(np.mean((residual / self.y_vals) ** 2))

        if self.args_dict['verbose']:
            final_params_joined = [i.tolist() for j in self.final_params for i in j]
            initial_params_joined = [i.tolist() for j in self.initial_params for i in j]
            peaks_only_final = np.array([param[:2] for param in final_params_joined])
            peaks_only_initial = np.array([param[:2] for param in initial_params_joined])
            print(f'Fitter Function – Iteration no. {self.iterations_distr}')
            print('--------------------------')
            print(f'Input parameters  : {params}')
            print(f'RMSD              : {self.rmsd}')
            print(f'Approximators     : {self.distributions}')
            print(f'Reshuffled bounds : {bounds_numbered}')
            print(f'Initial parameters : ')
            print(initial_params_joined)
            print(f'Final parameters : ')
            print(final_params_joined)
            print()
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))
            plt.title('Fitting Results')
            
            axs[0].plot(self.x_vals, self.y_vals, label='Reference spectrum')
            axs[0].plot(self.x_vals, self.initial_approximation.sum(axis=0), label=f'Initial spectrum at {self.iterations_distr}. iteration')
            axs[0].plot(self.final_approximation[:,0], self.final_approximation[:,1], label=f'Final spectrum at {self.iterations_distr}. iteration')
            axs[0].legend()
            
            axs[1].plot(self.x_vals, self.y_vals, label='Reference spectrum')
            axs[1].plot(self.x_vals, self.initial_approximation.sum(axis=0), label=f'Initial spectrum at {self.iterations_distr}. iteration')
            axs[1].plot(self.final_approximation[:,0], self.final_approximation[:,1], label=f'Final spectrum at {self.iterations_distr}. iteration')
            axs[1].plot(peaks_only_initial[:,0], peaks_only_initial[:,1], label="Initial Peaks", color='k', marker='x', ls='None')
            axs[1].plot(peaks_only_final[:,0], peaks_only_final[:,1], label="Final Peaks", color='r', marker='x', ls='None')
            axs[1].legend()
            
            plt.tight_layout()
            plt.show()
        
        return residual
        

class ComplexFitterLinearCombination():
    
    def __init__(
        self, 
        data: np.ndarray, 
        peaks: np.ndarray, 
        spec_bounds: np.ndarray, 
        peak_rtol: Optional[float] = 5e-02, 
        max_iter: Optional[int] = 100,
        residual: Optional[str] = 'default',
        verbose: bool = False
    ):

        self.verbose = verbose
        self.residual_type = residual
        
        self.x_vals = data[:,0]
        self.y_vals = self.min_max_scaling(data[:,1])
        mask_outside = (self.x_vals >= 400) & (self.x_vals <= 3500)
        self.x_vals = self.x_vals[mask_outside]
        self.y_vals = self.y_vals[mask_outside]
        # final_approximation = np.array([x_vals, np.zeros_like(y_vals)]).T
                        
        # initial parameters
        self.centers = peaks[:,0]
        self.amplitudes = self.min_max_scaling(peaks[:,1], ref_data=data[:,1])
        self.lorentz_widths = np.full_like(self.amplitudes, 15)
        self.gauss_widths = np.full_like(self.amplitudes, 15)
        self.weights_init = np.zeros(3)
        # self.spec_bounds = spec_bounds
        self.rmsd = 0
        self.iterations = 1

        gaussian_params = np.array([self.centers, self.amplitudes, self.gauss_widths]).T.flatten()
        lorentzian_params = np.array([self.centers, self.amplitudes, self.lorentz_widths]).T.flatten()
        voigt_params = np.array([self.centers, self.amplitudes, self.gauss_widths, self.lorentz_widths]).T.flatten()
        self.init_params = np.concatenate([self.weights_init, gaussian_params, lorentzian_params, voigt_params])

        center_lb = []
        center_ub = []
        for i in range(spec_bounds.shape[0]-1):
            x_lb = spec_bounds[i]
            x_ub = spec_bounds[i+1]
            mask = (self.centers >= x_lb) & (self.centers < x_ub)
            allowed_dev = (x_ub - x_lb) * peak_rtol
            centers_i = self.centers[mask]
            center_lb_i = centers_i - allowed_dev
            center_ub_i = centers_i + allowed_dev
            center_lb.append(center_lb_i)
            center_ub.append(center_ub_i)
        
        center_lb = np.concatenate(center_lb)
        center_ub = np.concatenate(center_ub)
        amplitude_lb = np.full_like(self.amplitudes, 1e-10)
        amplitude_ub = np.full_like(self.amplitudes, self.amplitudes.max() * 2)
        zero_bound = np.zeros_like(self.centers)
        inf_bound = np.full_like(self.centers, np.inf)

        gaussian_lorentzian_lb = np.array([center_lb, amplitude_lb, zero_bound]).T.flatten()
        voigt_lb = np.array([center_lb, amplitude_lb, zero_bound, zero_bound]).T.flatten()
        gaussian_lorentzian_ub = np.array([center_ub, amplitude_ub, inf_bound]).T.flatten()
        voigt_ub = np.array([center_ub, amplitude_ub, inf_bound, inf_bound]).T.flatten()
        weghts_lb = np.zeros_like(self.weights_init)
        weights_ub = np.full_like(self.weights_init, np.inf)
        self.bounds = (np.concatenate([weghts_lb, gaussian_lorentzian_lb, gaussian_lorentzian_lb, voigt_lb]), np.concatenate([weights_ub, gaussian_lorentzian_ub, gaussian_lorentzian_ub, voigt_ub]))
        
        self.approximator(max_iter)

    def max_scaling(self, data, ref_data=None):
        if ref_data is None:
            rescaled = data / data.max()
        else:
            rescaled = data / ref_data.max()
        return rescaled

    def approximator(self, max_iter):
        
        self.params = least_squares(self.residual_fun,
                            self.init_params, bounds=self.bounds,
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x

        weights, gaussian, lorentzian, voigt = self.unpack_params(self.params)
        self.results = self.weighted_sum(self.x_vals, weights, gaussian, lorentzian, voigt)
        
        return None

    def unpack_params(self, params):
        num_peaks = self.centers.shape[0]
        weights = softmax(params[:3])
        gaussian = params[3:3+(num_peaks*3)]
        lorentzian = params[3+(num_peaks*3):3+(num_peaks*3)+(num_peaks*3)]
        voigt = params[3+(num_peaks*3)+(num_peaks*3):]

        return weights, gaussian, lorentzian, voigt
    
    def gaussian(self, x, center, amplitude, gauss_width):
        # amplitude = amplitude * (-1.0)
        sigma = gauss_width / np.sqrt(2 * np.log(2))
        return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

    def gaussian_sum(self, x, params):
        params = params.flatten().tolist()
        params = [params[i:i + 3] for i in range(0, len(params), 3)]
        decompositions = [self.gaussian(x, center, amp, sigma) for center, amp, sigma in params]
        return np.sum(decompositions, axis=0)

    def lorentzian(self, x, center, amplitude, lorentz_width):
        gamma = lorentz_width / 2
        return (amplitude * gamma ** 2) / (gamma ** 2 + (x - center) ** 2)

    def lorentzian_sum(self, x, params):
        params = params.tolist()
        params = [params[i:i + 3] for i in range(0, len(params), 3)]
        decompositions = [self.lorentzian(x, centre, amp, gamma) for centre, amp, gamma in params]
        return np.sum(decompositions, axis=0)

    def voigt(self, x, center, amplitude, gauss_width, lorentz_width):
        sigma = gauss_width / np.sqrt(2 * np.log(2))
        gamma = lorentz_width / 2.0
        
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2) + 1e-20)
        real_part = np.real(wofz(z))
        norm = sigma * np.sqrt(2 * np.pi)
        profile = amplitude * real_part / norm
        return profile

    def voigt_sum(self, x, params):
        params = params.tolist()
        params = [params[i:i + 4] for i in range(0, len(params), 4)]
        decompositions = [self.voigt(x, centre, amp, gw, lw) for centre, amp, gw, lw in params]
        return np.sum(decompositions, axis=0)

    def weighted_sum(self, x_vals, weights, gaussian, lorentzian, voigt):
        return weights[0] * self.gaussian_sum(x_vals, gaussian) + weights[1] * self.lorentzian_sum(x_vals, lorentzian) + weights[2] * self.voigt_sum(x_vals, voigt)
        
    def residual_fun(self, params):
        weights, gaussian, lorentzian, voigt = self.unpack_params(params)

        fit = self.weighted_sum(self.x_vals, weights, gaussian, lorentzian, voigt)

        residual = self.y_vals - fit if self.residual_type == 'default' else np.log10(self.y_vals) - np.log10(fit)
        self.rmsd = np.sqrt(np.mean((residual / self.y_vals) ** 2))

        if self.verbose:
            print(f'Fitter Function – Iteration no. {self.iterations}')
            print('--------------------------')
            print(f'Weights:')
            print(f'Gaussian : {weights[0]} | Lorentzian : {weights[1]} | Voigt : {weights[2]}')
            print(f'RMSD              : {self.rmsd}')
            print()
            fig = plt.figure(figsize=(10, 5))
            plt.title('Fitting Results')
            plt.plot(self.x_vals, self.y_vals, label='Reference spectrum')
            plt.plot(self.x_vals, fit, label=f'Fitted spectrum at {self.iterations}. iteration')
            plt.xlabel('Wavenumbers $[cm^{-1}]$')
            plt.ylabel('Min-max scaled intensity')
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.iterations += 1
        
        return residual
