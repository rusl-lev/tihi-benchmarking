from scipy.optimize import least_squares
from scipy.special import wofz
import numpy as np

class GaussianFitter():
    def __init__(self, InterpolatedData, peaks,
                 max_iter=100):
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
        self.x_vals = InterpolatedData.x_val
        self.y_vals = InterpolatedData.y_val
        
        # initial parameters
        self.centers = self.x_vals[peaks]
        self.amplitudes = self.y_vals[peaks]
        self.sigmas = np.random.rand(*self.amplitudes.shape)
        self.params = np.array([self.centers, self.amplitudes,self.sigmas]).T  
        self.start_params = self.params.flatten().tolist()
        self.decompositions = []
        # result to output
        self.result = None
                
        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        """
        Perform Gaussian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        self.params = least_squares(self.residual,
                            self.start_params, args=(self.x_vals, self.y_vals),
                            bounds=(-np.max(self.x_vals),
                                    np.max(self.x_vals)),
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        print(self.params)
        print("the error for this run is: ", np.mean(self.residual(self.params, self.x_vals, self.y_vals)))

        self.results = np.array([self.gaussian_sum(x, self.params) for x in self.x_vals])
        error = np.mean(np.abs(self.y_vals - self.results))
        
        return error
    
    def gaussian(self, x, center, amplitude, sigma):
        """
        Calculate a Gaussian function.
        
        :param x (ndarray) : X values.
        :param center (float) : Center of the Gaussian function.
        :param amplitude (float) : Amplitude of the Gaussian function.
        :param sigma (float) : Standard deviation of the Gaussian function.
        :return (ndarray) : Calculated Gaussian function values.
        """
        amplitude = amplitude * (-1.0)
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
    def __init__(self, data, peaks, spec_bounds, peak_rtol,
                 max_iter=100):
        self.x_vals = data[:,0]
        self.y_vals = data[:,1]
        
        # initial parameters
        self.peak_position_tolerance = peak_rtol
        self.bounds = spec_bounds
        self.centers = peaks[:,0]
        self.amplitudes = peaks[:,1]
        self.gammas = [1]*len(self.centers)
        self.params = np.array([self.centers, self.amplitudes, self.gammas]).T
        # self.start_params = self.params.flatten().tolist()
        self.decompositions = []
        # self.output_params = np.zeros(shape=(self.centers.shape[0], 3))
        self.output_params = []
        self.approximation_results = []
        print(f'Shape of the bounds array: ', self.bounds.shape)
        print(f'Bounds array: ', self.bounds)
        print(f'X values: {self.x_vals}')

        for i in range(self.bounds.shape[0]-1):
            x_ub = self.bounds[i+1]
            x_lb = self.bounds[i]
            print(f'Lower bound: {x_lb}; upper bound: {x_ub}')
            allowed_dev = (x_ub - x_lb) * self.peak_position_tolerance
            peak = self.centers[i]
            peak_deviation_bound = ([(peak - allowed_dev), 0, 0], [(peak + allowed_dev), np.inf, np.inf])
            mask = (self.x_vals >= x_lb) & (self.x_vals <= x_ub)
            x_masked = self.x_vals[mask]
            y_masked = self.y_vals[mask]
            print(f'X masked: {x_masked}')
            print(f'Y masked: {y_masked}')
            approx, params = self.approximator(max_iter, self.params[i], peak_deviation_bound, x_masked, y_masked)
            print(f'Paramerers in {i}th iteration: ', params)
            print(f'Lorentzian approximation in {i}th iteration', approx)
            # self.output_params[i] = params
            self.output_params.append(params)
            self.approximation_results.append(approx)
        self.output_params = np.array(self.output_params)
        self.approximation_results = np.array(self.approximation_results)
        
    def approximator(self, max_iter, start_params, bounds, x_vals, y_vals):
        """
        Perform Lorentzian fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        parameters = least_squares(self.residual,
                            start_params, args=(x_vals, y_vals),
                            bounds=bounds,
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        
        # params = np.array([parameters[i:i + 3] for i in range(0, len(parameters), 3)])
        # print(self.params)
        # print("the error for this run is: ", np.mean(self.residual(self.params, self.x_vals, self.y_vals)))

        approximation_results = np.array([self.lorentzian_sum(x, parameters) for x in x_vals])
        # error = np.mean(np.abs(self.y_vals - self.results))
        
        return approximation_results, parameters
    
    def lorentzian(self, x, center, amplitude, gamma):
        """
        Calculate a Lorentzian function.
        
        :param x (ndarray) : X values.
        :param center (float) : Center of the Lorentzian function.
        :param amplitude (float) : Amplitude of the Lorentzian function.
        :param gamma (float) : Full width at half maximum (FWHM) of the Lorentzian function.
        :return (ndarray) : Calculated Lorentzian function values.
        """
        # amplitude = amplitude * (-1.0)
        gamma /= 2
        return amplitude * (gamma / np.pi) / ((x - center) ** 2 + gamma ** 2)

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
    
class VoigtFitter():
    def __init__(self, InterpolatedData, peaks, max_iter=50):
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
        self.x_vals = InterpolatedData.x_val
        self.y_vals = InterpolatedData.y_val
        
        # initial parameters
        self.centers = self.x_vals[peaks]
        self.amplitudes = self.y_vals[peaks]
        self.gauss_widths = np.ones_like(self.amplitudes) #np.random.rand(*self.amplitudes.shape)
        self.lorentz_widths = np.ones_like(self.amplitudes) #np.random.rand(*self.amplitudes.shape)
        self.params = np.array([self.centers, self.amplitudes, self.gauss_widths, self.lorentz_widths]).T
        self.start_params = self.params.flatten().tolist()
        num_dists = len(self.centers)
        self.decompositions = []
        
        self.approximator(max_iter)
        
    def approximator(self, max_iter):
        """
        Perform Voigt fitting using least squares optimization.
        
        :param max_iter (int) : Maximum number of iterations for fitting.
        :return error (float) : Mean absolute error of the fitting.
        :Notes : Uses soft L1 loss and bounds parameters to constrain optimization.
        """
        self.params = least_squares(self.residual,
                            self.start_params, args=(self.x_vals, self.y_vals),
                            bounds=(-np.max(self.x_vals), np.max(self.x_vals)),
                            ftol=1e-9, xtol=1e-9, loss='soft_l1',
                            f_scale=0.1, max_nfev=max_iter).x
        print(self.params)
        print("the error for this run is: ", np.mean(self.residual(self.params, self.x_vals, self.y_vals)))

        self.results = np.array([self.voigt_sum(x, self.params) for x in self.x_vals])
        error = np.mean(np.abs(self.y_vals - self.results))
        
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
        gamma = lorentz_width
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
