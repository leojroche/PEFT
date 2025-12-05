# Léo J. Roche (leoroche2@gmail.comm)
# 03/12/2025

import numpy, os, pandas, re, json
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
from nptdms import TdmsFile
from tools import *


def linear_function(x, slope, intercept):
    return slope * numpy.array(x) + intercept

# --------------- single Voigt profile functions with different background functions
# https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Stellar_Atmospheres_(Tatum)/10%3A_Line_Profiles/10.04%3A_Combination_of_Profiles#:~:text=Convolving%20a%20Lorentzian%20Function%20with%20a%20Gaussian%20Function,-Let%20us%20now&text=or-,V(x)%3D1g%E2%88%9Aln2%CF%803,is%20called%20a%20Voigt%20profile.
# FWHM of a Voigt Profile at (https://en.wikipedia.org/wiki/Voigt_profile) 
def voigt_profile_func(x, x0, a, sigma, gamma, b): # Voigt profile function plus a constant
    return (a-b)*voigt_profile( x-x0, sigma, gamma)/voigt_profile(0, sigma, gamma) + b

def voigt_profile_plus_linear_func(x, x0, a, sigma, gamma, slope, intercept): # Voigt profile function plus a linear function
    # FWHM of a Voigt Profile at (https://en.wikipedia.org/wiki/Voigt_profile) 
    b = slope * numpy.array(x) + intercept
    return (a-b)*voigt_profile( x-x0, sigma, gamma)/voigt_profile(0, sigma, gamma) + b

def voigt_profile_no_const(x, x0, a, sigma, gamma):
    return a*voigt_profile( x-x0, sigma, gamma)/voigt_profile(0, sigma, gamma)



# --------------- multiple Voigt profile functions
def double_voigt_profile_func(x, x0_1, a_1, sigma_1, gamma_1, x0_2, a_2, sigma_2, gamma_2, b):
    return (a_1-b)*voigt_profile( x-x0_1, sigma_1, gamma_1)/voigt_profile(0, sigma_1, gamma_1) + (a_2-b)*voigt_profile( x-x0_2, sigma_2, gamma_2)/voigt_profile(0, sigma_2, gamma_2) + b 
    
def _4_voigt_function(x, x0_1, a_1, sigma_1, gamma_1, x0_2, a_2, sigma_2, gamma_2, x0_3, a_3, sigma_3, gamma_3, x0_4, a_4, sigma_4, gamma_4):
    return voigt_profile_plus_linear_func(x, x0_1, a_1, sigma_1, gamma_1, slope=0, intercept=0)  + voigt_profile_plus_linear_func(x, x0_2, a_2, sigma_2, gamma_2, slope=0, intercept=0)  + voigt_profile_plus_linear_func(x, x0_3, a_3, sigma_3, gamma_3, slope=0, intercept=0) + voigt_profile_plus_linear_func(x, x0_4, a_4, sigma_4, gamma_4, slope=0, intercept=0)

def _log_4_voigt_function(x, x0_1, a_1, sigma_1, gamma_1, x0_2, a_2, sigma_2, gamma_2, x0_3, a_3, sigma_3, gamma_3, x0_4, a_4, sigma_4, gamma_4):
    return numpy.log(voigt_profile_plus_linear_func(x, x0_1, a_1, sigma_1, gamma_1, slope=0, intercept=0)  + voigt_profile_plus_linear_func(x, x0_2, a_2, sigma_2, gamma_2, slope=0, intercept=0)  + voigt_profile_plus_linear_func(x, x0_3, a_3, sigma_3, gamma_3, slope=0, intercept=0) + voigt_profile_plus_linear_func(x, x0_4, a_4, sigma_4, gamma_4, slope=0, intercept=0))


# ---------------- gaussian functions 

def gaussian_function(x, x0, a, sigma):
    return a * numpy.exp(-(x-x0)**2 / (2*sigma**2))

def gaussian_function_plus_constant(x, x0, a, sigma, b):
    return gaussian_function(x, x0, a, sigma) + b

def gaussian_function_plus_linear_func(x, x0, a, sigma, slope, intercept):
    return gaussian_function(x, x0, a, sigma) + slope * x + intercept

def double_gaussian_function(x, x0_1, a_1, sigma_1, x0_2, a_2, sigma_2, b):
    return gaussian_function(x, x0_1, a_1, sigma_1) + gaussian_function(x, x0_2, a_2, sigma_2) + b


class Spectrum:
    """Contains all the information about a single spectrum"""
    
    wavelengths: numpy.ndarray
    "Wavelengths of the spectrum [nm]"

    intensities: numpy.ndarray
    "Intensities of the spectrum [a.u.]"

    x : float
    "Excitation power or current"

    fits: list[dict]
    "List of dictionaries containing to fit results"

    def __init__(self, FILEPATH, wavelengths = None, intensities = None, x = None, OD = None):
        """
        Load `Spectrum` either by reading a measurement file or by passing the data as arguments.
        The reading of new file formats can be implemented here where the attributes **wavelengths**, **intensities** and **x** have to be assigned.
        The data file format has to be specified in the 'measurements_infos.json' file.

        :param FILEPATH: (str) filepath of the measurement file to load.
        :param wavelengths: (numpy.ndarray) [nm] wavelengths of the acquired spectrum
        :param intensities: (numpy.ndarray) [nm] intensities of the acquired spectrum
        :param x: (float) power or current
        :param OD: (float)  neutral OD filter value

        :return: None 
        """
        self.FILEPATH = FILEPATH
        self.peaks = []
        self.fits = []

        # find measurement_infos file
        meas_infos_filepath = find_measurements_infos_file(FILEPATH)
        with open(meas_infos_filepath, 'r') as file:
            self.meas_infos = json.load(file)
        serie_fileformat = self.meas_infos['serie_fileformat']
    
        if x==None : # means data of a single spectrum must be read from FILEPATH

            # -------------------------------------------
            # Here add use cases to read new formats 
            # Example :
            # "
            # if serie_fileformat=='my_new_format':
            #   data_file = pandas.('my_txt_measurement_file.txt', 'rb')
            #   self.wavelengths = data_file['WAVELENGTH_NM']
            #   self.intensities = data_file['COUNTS']
            #   self.x = data_file['POWER']
            # "
            # -------------------------------------------
            
            if serie_fileformat=='qlab2' or serie_fileformat=='qlab4': # loads spectrum exported by LabVIEW on the computer in qlab 2
                df = pandas.read_csv(self.FILEPATH, sep='\t', header=None)
                self.wavelengths = df[1] # nm
                self.intensities = df[2] # a.u.
                self.x = 0
                with open(self.FILEPATH + 'LogFile', 'r', encoding='unicode_escape') as file:
                    log_file = file.read()
                self.OD = float(re.search(r'OD Filter = "([+-]?([0-9]*[.])?[0-9]+)"', log_file).group(1)) 
                self.x = float(re.search(r'Power = (\d+.\d+)', log_file).group(1)) * 1e3 # mW

            if serie_fileformat=='qlab4_rt_agilent':
                df = pandas.read_csv(self.FILEPATH, sep='\t', header=None)
                self.wavelengths = df[1] # nm
                self.intensities = df[2] # a.u.
                self.x = 0

            elif serie_fileformat=='qlab5_picoprobe': # concatenating all the spectra containing in folder self.FILEPATH (stepandglue)
                if os.path.isdir(self.FILEPATH):
                    dfs = []
                    for filename in os.listdir(self.FILEPATH):
                        if filename.endswith('_'):
                            file_path = os.path.join(self.FILEPATH, filename)
                            df = pandas.read_csv(file_path, header=None, sep='\t')  # Adjust the separator if needed
                            i_min = 478
                            i_max = -200
                            df = df.iloc[i_min:i_max, :]
                            dfs.append(df)
                    df = pandas.concat(dfs, ignore_index=True)
                    df = df.sort_values(by=[0])
                    self.wavelengths = df.iloc[:, 0]
                    self.intensities = df.iloc[:, 1]
                    self.x = None
                    self.OD =  None
                
                if os.path.isfile(self.FILEPATH): # load single spectrum from .tdms file
                    tdms_file = TdmsFile.read(FILEPATH)
                    self.wavelengths = tdms_file['Spectrum: SpecAxis[0-2]']['Data'].data
                    self.intensities = tdms_file['Spectrum: Intensity[0-1]']['Data'].data
                    self.x = None
                    self.OD =  None
                        

        else : # the spectrum is being created by a 'Series'
            self.wavelengths = wavelengths
            self.intensities = intensities
            self.x = x
            self.OD = OD


    def scale_intensity(self, factor):
        """ 
        Multiply all intensity values by **factor**. Can be used the manually stitch different measurement `Serie`s.

        :param factor: (float) multiplying factor
        :return: self
        """

        self.intensities = [factor * intensity for intensity in self.intensities]
        return self

    def normalize_intensity(self):
        """ 
        Normalize the intensities.

        :return: self
        """
        self.intensities -= numpy.min(self.intensities)
        self.intensities /= numpy.max(self.intensities)
        return self

    def apply_OD(self):
        """ 
        Calculate the number of counts there would be if the OD specified by **OD** was removed.
        """
        self.scale_intensity(10**(self.OD))
        self.OD = 0

    def remove(self, lambda_min = None, lambda_max = None):
        """
        Remove the data points of a specific region of the spectrum specified by [**lambda_min**, **lambda_max**]

        :param lambda_min: (float) [nm] lower bound of the spectral range to remove. Can be set to None.
        :param lambda_max: (float) [nm] upper bound of the spectral range to remove. Can be set to None.

        :return: self
        """
        if lambda_min == None:
            lambda_min = - numpy.infty
        if lambda_max == None:
            lambda_max =  numpy.infty
        df = pandas.DataFrame({'wavelengths':self.wavelengths, 'intensities':self.intensities})
        df = df[(df['wavelengths']<=lambda_min) | (df['wavelengths']>=lambda_max)]
        self.wavelengths = df['wavelengths'].values
        self.intensities = df['intensities'].values
        return self


    def sum_intensities(self, lambda_min = None, lambda_max = None):
        """ 
        Return the sum of the intensities of the whole spectrum or only of the spectral region specified by [**lambda_min**, **lambda_max**]

        :param lambda_min: (float) [nm] lower bound of the spectral window to integrate
        :param lambda_max: (float) [nm] upper bound of the spectral window to integrate

        :return: sum
        """
        if lambda_min == None:
            lambda_min = - numpy.infty
        if lambda_max == None:
            lambda_max = numpy.infty

        df = pandas.DataFrame({'wavelengths':self.wavelengths, 'intensities':self.intensities})
        df = df[(df['wavelengths']>=lambda_min) & (df['wavelengths']<=lambda_max)]

        return numpy.sum(df['intensities'])


    def plot(self, lambda_min = -numpy.infty, lambda_max = numpy.infty, y_min = None, y_max = None, show=False, save=False):
        """
        Plot spectrum.

        :param lambda_min: (float) [nm] minimum wavelength to be plotted 
        :param lambda_max: (float) [nm] maximum wavelength to be plotted 
        :param y_min: (float) [a.u.] minimum value of the intensity shown 
        :param y_max: (float) [a.u.] maximum value of the intensity shown 
        :param show: (bool) if true, show the generated plot
        :param save: (bool) if true save the generated plots in the same folder as the measurement file

        :return: self
        """
        color='dodgerblue'
        plt.figure()
        d = pandas.DataFrame({'wavelengths':self.wavelengths, 'intensities':self.intensities})
        d = d[(d['wavelengths']>=lambda_min) & (d['wavelengths']<=lambda_max)]
        plt.plot(d['wavelengths'], d['intensities'], '.', alpha=1.0, color=color, markersize=1)
        plt.plot(d['wavelengths'], d['intensities'], '--', alpha=0.3, color=color)
        if self.peaks != []:
            for peak in self.peaks:
                i0 = numpy.where(self.wavelengths == peak)[0][0]
                plt.plot(peak, self.intensities[i0], "xr")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.grid(True)
        title = 'Spectrum\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
        plt.title(title, fontsize=7)
        if y_min != None:
            plt.ylim(bottom=y_min)
        if y_max != None:
            plt.ylim(top=y_max)
        if save:
            print(self.FILEPATH+'.png')
            plt.savefig(self.FILEPATH+'.png')
        if show:
            plt.show()
        return self


    def plot_with(self, other_spectra=[], labels=[], vertical_offset=0, lambda_min = - numpy.infty, lambda_max = numpy.infty, show=False):
        """
        Show the spectrum with other spectra on the same plot.

        :param other_spectra: (list of `Spectrum`) list of spectra to be plotted
        :param vertical_offset: (int) [a.u.] value to add to the intensity of the spectra in order to stack them up vertically on the same plot 
        :param lambda_min: (float) [nm] minimum wavelength to be plotted 
        :param lambda_max: (float) [nm] maximum wavelength to be plotted 
        :param show: (bool) if true shows the generated plot 

        :return: self
        """
        colors = cm.gist_rainbow(numpy.linspace(0.05, 0.68, len(other_spectra)+1))
        plt.figure()
        d = pandas.DataFrame({'wavelengths':self.wavelengths, 'intensities':self.intensities})
        d = d[(d['wavelengths']>=lambda_min) & (d['wavelengths']<=lambda_max)]
        if labels==[]:
            label=''
        else: 
            label = labels[0]
        plt.plot(d['wavelengths'], d['intensities'], '.', alpha=1.0, color=colors[0], label=label)
        plt.plot(d['wavelengths'], d['intensities'], '--', alpha=0.3, color=colors[0])
        for i in range(1, len(other_spectra)+1):
            s = other_spectra[i-1]
            if labels==[]:
                label=''
            else: 
                label = labels[i]
            d = pandas.DataFrame({'wavelengths':s.wavelengths, 'intensities':s.intensities})
            d = d[(d['wavelengths']>=lambda_min) & (d['wavelengths']<=lambda_max)]
            plt.plot(d['wavelengths'], d['intensities'] + i*vertical_offset, '.', alpha=1.0, color=colors[i], label=label)
            plt.plot(d['wavelengths'], d['intensities'] + i*vertical_offset, '--', alpha=0.3, color=colors[i])

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.grid(True)
        plt.legend()
        if show:
            plt.show()
        
        return self
        
    def fit_gauss(self, lambda0_guess, delta_lambda=.5, sigma_guess = .01, background_type = 'linear', bounds = None, plot = False):
        """
        Fit spectrum with a Gaussian function

        Parameters : 
        :param lambda0_guess: (float) [nm] guess value of central position of the Gaussian function
        :param delta_lambda: (float) [nm] range of wavelengths centered around lambda0_guess used for the fitting
        :param sigma_guess: (float) guess value of the standard deviation of the Gaussian function
        :param background_type: (str)  defines the type of background function to fit. Accepted values are 
           - "none"    : Gaussian function only
           - "const"   : Gaussian function with constant background
           - "linear"  : Gaussian function with linear function
        :param bounds: (dict) dictionary containing the limit values that the function parameters can reach during the fitting procedure. The available keys are
           - "sigma_min"
           - "sigma_max"
        :param plot: (bool) : if true, plot the spectrum and the fitted function

        :return: dictonary of result if successful, None if failed
        """

        fit_function_color = 'orangered'
        try:
                # preparation of the data to fit :
                i0_guess = (numpy.abs(self.wavelengths - lambda0_guess)).argmin()
                i0_min =  (numpy.abs(self.wavelengths - (lambda0_guess - delta_lambda/2.))).argmin()
                i0_max =  (numpy.abs(self.wavelengths - (lambda0_guess + delta_lambda/2.))).argmin()
                # crop the spectrum to do the fit 
                xs = numpy.array(self.wavelengths[i0_min:i0_max]) # wavelengths (nm)
                ys = numpy.array(self.intensities[i0_min:i0_max]) # intensities (a.u.)
                # default bounds if bounds parameters is None
                bound_min = [min(xs), 0, 0]
                bound_max = [max(xs), numpy.infty, numpy.infty]
                # apply bounds from the 'bounds' variables
                if bounds != None:
                    if 'sigma_min' in bounds:
                        bound_min[2] = bounds['sigma_min']
                    if 'sigma_max' in bounds:
                        bound_max[2] = bounds['sigma_max']
                

                if background_type=='linear': # fits a gaussian function + a linear function on the data
                    slope_guess = (ys[-1] - ys[0])/(xs[-1] - ys[0])
                    intercept_guess = ys[-1] - slope_guess*xs[-1]
                    a_guess = self.intensities[i0_guess]

                    # default bounds if bounds parameters is None
                    bound_min = [min(xs), 0, 0, -numpy.infty, -numpy.infty]
                    bound_max = [max(xs), numpy.infty, numpy.infty, numpy.infty, numpy.infty]
                    # apply bounds from the 'bounds' variables
                    if bounds != None:
                        if 'sigma_min' in bounds:
                            bound_min[2] = bounds['sigma_min']
                        if 'sigma_max' in bounds:
                            bound_max[2] = bounds['sigma_max']

                    p0=[lambda0_guess, a_guess, sigma_guess, slope_guess, intercept_guess]
                    popt, pconv = curve_fit(gaussian_function_plus_linear_func, xs, ys, p0=p0, bounds=(bound_min, bound_max))
                    lambda0 = float(popt[0])
                    a = float(popt[1])
                    sigma = float(numpy.abs(popt[2]))
                    slope = float(popt[3])
                    intercept = float(popt[4])
                    f_g = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma # FWHM of a gaussian function (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
                    q = lambda0/f_g # Q-factor
                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'gaussian_function',
                                        'background_type' : background_type,
                                        'lambda0_guess' : lambda0_guess,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'slope_guess'      : slope_guess,
                                        'intercept_guess'      : intercept_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0, # central wavelength (nm)
                                        'a' : a, # amplitude of the Voigt profile (a.u.)
                                        'sigma' : sigma, 
                                        'slope' : slope,
                                        'intercept' : intercept,
                                        'f_g' : f_g,
                                        'q' : float(q),
                    }
                    self.fits.append(fit_result)

                    if plot:
                        fig, ax = plt.subplots()
                        # plot data
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        # plot fitted linear function taking into account the linear background
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001)
                        intensities_fit = linear_function(wavelength_fit, float(fit_result['slope']), float(fit_result['intercept']))
                        ax.plot(wavelength_fit, intensities_fit, color='mediumorchid', linestyle='--')
                        # plot the fitted function
                        intensities_fit = gaussian_function_plus_linear_func(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'], fit_result['slope'], fit_result['intercept'])
                        ax.plot(wavelength_fit, intensities_fit, label='Gaussian fit', zorder=-2, color=fit_function_color)
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Gaussian Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result

                elif background_type=='const': # fits a gaussian function + a constant on the data (b)
                    b_guess = 0
                    guess_intensity = self.intensities[i0_guess]
                    p0=[lambda0_guess, guess_intensity, sigma_guess, b_guess]
                    # default bounds if bounds parameters is None
                    bound_min = [min(xs), 0, 0, 0]
                    bound_max = [max(xs), numpy.infty, numpy.infty, numpy.infty]
                    # apply bounds from the 'bounds' variables
                    if bounds != None:
                        if 'sigma_min' in bounds:
                            bound_min[2] = bounds['sigma_min']
                        if 'sigma_max' in bounds:
                            bound_max[2] = bounds['sigma_max']

                    popt, pconv = curve_fit(gaussian_function_plus_constant, xs, ys, p0=p0, bounds=(bound_min, bound_max))
                    lambda0 = float(popt[0])
                    a = float(popt[1])
                    sigma = float(numpy.abs(popt[2]))
                    b = float(popt[3])
                    f_g = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma # FWHM of a gaussian function (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
                    q = lambda0/f_g
                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'gaussian_function',
                                        'background_type' : background_type,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'b_guess'      : b_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0,
                                        'a' : a,
                                        'sigma' : sigma,
                                        'b' : b,
                                        'f_g' : f_g,
                                        'q' : q,
                    }
                    self.fits.append(fit_result)

                    if plot:
                        fig, ax = plt.subplots()
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        # plot fitted constant background
                        ax.plot([min(xs), max(xs)], [b, b], color='mediumorchid', linestyle='--')
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001)
                        intensities_fit = gaussian_function_plus_constant(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'], fit_result['b'])
                        ax.plot(wavelength_fit, intensities_fit, label='Gaussian Fit', color='orangered')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Gaussian Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result

                elif background_type=='none': # fits a gaussian function without any constant
                    guess_intensity = self.intensities[i0_guess]
                    p0=[lambda0_guess, guess_intensity, sigma_guess]

                    # default bounds if bounds parameters is None
                    bound_min = [min(xs), 0, 0]
                    bound_max = [max(xs), numpy.infty, numpy.infty]
                    # apply bounds from the 'bounds' variables
                    if bounds != None:
                        if 'sigma_min' in bounds:
                            bound_min[2] = bounds['sigma_min']
                        if 'sigma_max' in bounds:
                            bound_max[2] = bounds['sigma_max']

                    popt, pconv = curve_fit(gaussian_function, xs, ys, p0=p0, bounds=(bound_min, bound_max))
                    lambda0 = popt[0]
                    a = popt[1]
                    sigma = numpy.abs(popt[2])
                    f_g = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma # FWHM of a gaussian function (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
                    q = lambda0/f_g

                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'gaussian_function',
                                        'background_type' : background_type,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0,
                                        'a' : a,
                                        'sigma' : sigma,
                                        'f_g' : f_g,
                                        'q' : q,
                    }
                    self.fits.append(fit_result)

                    if plot:
                        fig, ax = plt.subplots()
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        # plot fitted
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001)
                        intensities_fit = gaussian_function(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'])
                        ax.plot(wavelength_fit, intensities_fit, label='Gaussian Fit', color='orangered')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Gaussian Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result
        
        except RuntimeError:
            print('x = ' + str(self.x) + ' : Gaussian fit failed : scipy.optimize.curve_fit() failed')
            return None

        
    def fit_2_gauss(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess,  x0_2_guess, a_2_guess, sigma_2_guess, b_guess, bounds = None, plot=False, lambda_min_plot = None,  lambda_max_plot = None):
        """
        Fit 2 Gaussian functions on the spectrum

        Parameters : 
        :param lambda_min_fit: (float) [nm] : minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] : maximal wavelength used to fit the function
        :param x0_i_guess: (float) [nm] guess value of the position of the i-th peak 
        :param a_i_guess: (float) [nm] guess value of the amplitude of the i-th peak 
        :param sigma_i_guess: (float) sigma parameter of i-th Gaussian peak
        :param bounds: (dict) dictionary containing constraints of the fitted parameters. Available constraints are
           - "x0_1_min" (float) 
           - "x0_2_min" (float) 
           - "x0_1_max" (float) 
           - "x0_2_max" (float) 
           - "sigma_1_min" (float) 
           - "sigma_2_min" (float) 
        :param plot: (bool) if True, plots the data with the fitted function
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 

        :return: 
        - dict of result if successful
        - None if failed
        """
        # plot parameters
        color_data ='dodgerblue'
        ansatz_color = 'orange'
        gaussian_1_color = 'mediumseagreen'
        gaussian_2_color = 'crimson'
        alpha_individual_voigt_profile_ansatz = 0.7
        color_fitting_range = 'slategrey'
        alpha_fitting_range = 0.1
        color_fitted_function = 'lime'

        if lambda_min_plot == None:
            lambda_min_plot = lambda_min_fit
        if lambda_max_plot == None:
            lambda_max_plot = lambda_max_fit

        df_to_fit = pandas.DataFrame({'xs':self.wavelengths, 'ys':self.intensities})
        df_to_fit = df_to_fit[(df_to_fit['xs'] >= lambda_min_fit) & (df_to_fit['xs'] <= lambda_max_fit)]
        normalization_factor = numpy.max(df_to_fit['ys'])
        df_to_fit['ys'] /= normalization_factor
        df_to_plot = pandas.DataFrame({'xs':self.wavelengths, 'ys':self.intensities})
        df_to_plot = df_to_plot[(df_to_plot['xs'] >= lambda_min_plot) & (df_to_plot['xs'] <= lambda_max_plot)]
        p0 = [  x0_1_guess, a_1_guess,  sigma_1_guess,   # Gaussian 1
                x0_2_guess, a_2_guess,  sigma_2_guess,   # Gaussian 2
                b_guess ] # const background
        try : 
            a_1_guess_min = 1e-2 / normalization_factor
            a_2_guess_min = 1e-2 / normalization_factor
            bound_min = [lambda_min_fit, a_1_guess_min, - numpy.infty, 
                         lambda_min_fit, a_2_guess_min, - numpy.infty, 
                         0]
            a_1_guess_max = 2 * a_1_guess / normalization_factor
            a_2_guess_max = 2 * a_2_guess / normalization_factor
            bound_max = [lambda_max_fit, a_1_guess_max, 10, 
                         lambda_max_fit, a_2_guess_max, 10, 
                         numpy.infty]
            # apply bounds from the 'bounds' variables
            if bounds != None:
                # Gauss 1
                if 'x0_1_min' in bounds:
                    bound_min[0] = bounds['x0_1_min']
                if 'x0_1_max' in bounds:
                    bound_max[0] = bounds['x0_1_max']
                if 'sigma_1_min' in bounds:
                    bound_min[2] = bounds['sigma_1_min']
                # Gauss 2
                if 'x0_2_min' in bounds:
                    bound_min[3] = bounds['x0_2_min']
                if 'x0_2_max' in bounds:
                    bound_max[3] = bounds['x0_2_max']
                if 'sigma_2_min' in bounds:
                    bound_min[5] = bounds['sigma_2_min']

            p0[1] /= normalization_factor
            p0[4] /= normalization_factor
            p0[6] /= normalization_factor

            # fit 
            popt, pcov = curve_fit(double_gaussian_function, df_to_fit['xs'], df_to_fit['ys'], p0=p0, bounds=(bound_min, bound_max))
            popt[1] *= normalization_factor
            popt[4] *= normalization_factor
            popt[6] *= normalization_factor
            p0[1] *= normalization_factor
            p0[4] *= normalization_factor
            p0[6] *= normalization_factor
            
            # results Gauss 1
            x0_1 = popt[0]
            a_1 = popt[1] 
            sigma_1 = numpy.abs(popt[2])
            f_g_1 = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma_1 # FWHM of a gaussian function (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
            q_1 = x0_1/f_g_1
            # results Gauss 2
            x0_2 = popt[3]
            a_2 = popt[4]
            sigma_2 = popt[5]
            f_g_2 = 2 * numpy.sqrt(2 * numpy.log(2)) * sigma_2 # FWHM of a gaussian function (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
            q_2 = x0_2/f_g_2
            b = popt[6]
            # keep peak 1 to the left and peak 2 to the right
            if x0_1 > x0_2:
                x0_1, x0_2 =  x0_2, x0_1
                a_1, a_2 =  a_2, a_1
                sigma_1, sigma_2 =  sigma_2, sigma_1
                q_1, q_2 =  q_2, q_1

            # calculation of the maximum
            lambda_fit = numpy.arange(lambda_min_plot, lambda_max_plot, (lambda_max_plot - lambda_min_plot)/1e3)
            maximum = numpy.max(double_gaussian_function(lambda_fit, *popt))
            fit_results = {
                                'x' : self.x,
                                'fit_function' : '2_gaussian_functions',
                                'lambda_min_fit' : lambda_min_fit,
                                'lambda_max_fit' : lambda_max_fit,
                                'x0_1_guess' : x0_1_guess,
                                'a_1_guess' : a_1_guess,
                                'sigma_1_guess' : sigma_1_guess,
                                'x0_2_guess' : x0_2_guess,
                                'a_2_guess ' : a_2_guess,
                                'sigma_2_guess' : sigma_2_guess,
                                'b_guess'       : b_guess,
                                'x0_1'      : x0_1,
                                'a_1'      : a_1,
                                'sigma_1'      : sigma_1,
                                'f_g_1'      : f_g_1,
                                'q_1'      : q_1,
                                'x0_2'      : x0_2,
                                'a_2'      : a_2,
                                'sigma_2'      : sigma_2,
                                'f_g_2'      : f_g_2,
                                'q_2'      : q_2,
                                'b'        : b,
                                'maximum'   : maximum
            }
            self.fits.append(fit_results)

        except RuntimeError:
            print('x = ' + str(self.x) + ' : 2 Gaussian functions fit failed : scipy.optimize.curve_fit() failed')

            p0[1] *= normalization_factor
            p0[4] *= normalization_factor
            p0[6] *= normalization_factor

            if plot:
                title = 'Overview' + ',Fit Spectrum\n' + os.path.basename(self.FILEPATH) + ', x=' + "{:.4f}".format(self.x)
                fig, ax = plt.subplots()
                ax.set_title(title)
                # plot data
                df = pandas.DataFrame({'x':df_to_plot['xs'], 'y':df_to_plot['ys']})
                ax.plot(df_to_plot['xs'], df_to_plot['ys'], '.', alpha=1.0, color=color_data)
                ax.plot(df_to_plot['xs'], df_to_plot['ys'], '--', alpha=0.3, color=color_data)
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.grid(True, alpha = 0.5)
                ax.axvspan(xmin = lambda_min_fit, xmax = lambda_max_fit, color=color_fitting_range, alpha=alpha_fitting_range)
                lambda_fit = numpy.arange(lambda_min_plot, lambda_max_plot, (lambda_max_plot - lambda_min_plot)/1e3)
                # plot ansatz
                ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *p0[0:3], p0[-1]), ':', label='Gaussian 1 Ansatz', color=ansatz_color)
                ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *p0[3:6], p0[-1]), ':', label='Gaussian 2 Ansatz', color=ansatz_color)
                ansatz_function = double_gaussian_function(lambda_fit, *p0)
                ax.plot(lambda_fit, ansatz_function, '-', label='2 Gaussian Ansatz', color='orange')

            return None
        
        if plot:
            
            # ----------- plots
            fig, ax = plt.subplots()
            title = 'Fit Spectrum\n' + os.path.basename(self.FILEPATH) + ', x=' + "{:.4f}".format(self.x)
            # plot data
            ax.set_title(title)
            lambda_fit = numpy.arange(lambda_min_plot, lambda_max_plot, (lambda_max_plot - lambda_min_plot)/1e3)
            # plot ansatz functions
            ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *p0[0:3], p0[-1]), ':', label='Gaussian 1 Ansatz', color=gaussian_1_color, alpha = alpha_individual_voigt_profile_ansatz)
            ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *p0[3:6], p0[-1]), ':', label='Gaussian 2 Ansatz', color=gaussian_2_color, alpha = alpha_individual_voigt_profile_ansatz)
            ansatz_function = double_gaussian_function(lambda_fit, *p0)
            ax.plot(lambda_fit, ansatz_function, '-', label='2 Gaussian Ansatz', color=ansatz_color)
            # plot position of the fitted peaks
            ax.axvline(x0_1, linestyle='-', color=gaussian_1_color, alpha = 0.8)
            ax.axvline(x0_2, linestyle='-', color=gaussian_2_color, alpha = 0.8)
            # plot position peak bounds
            alpha_peak_bound_line = 1
            if bounds != None:
                # position Gaussian 1
                if 'x0_1_min' in bounds:
                    ax.axvline(bounds['x0_1_min'], linestyle='-', color='black', alpha = alpha_peak_bound_line)
                if 'x0_1_max' in bounds:
                    ax.axvline(bounds['x0_1_max'], linestyle='--', color='black', alpha = alpha_peak_bound_line)
                # position Gaussian 2
                if 'x0_2_min' in bounds:
                    ax.axvline(bounds['x0_2_min'], linestyle='-', color='black', alpha = alpha_peak_bound_line)
                if 'x0_2_max' in bounds:
                    ax.axvline(bounds['x0_2_max'], linestyle='--', color='black', alpha = alpha_peak_bound_line)

            # plot single Voigt profiles
            ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *popt[0:3], b), label='Gaussian 1', color=gaussian_1_color)
            ax.plot(lambda_fit, gaussian_function_plus_constant(lambda_fit, *popt[3:6], b), label='Gaussian 2', color=gaussian_2_color)
            ax.axvspan(xmin = lambda_min_fit, xmax = lambda_max_fit, color=color_fitting_range, alpha=alpha_fitting_range)
            # plot fitted function
            ax.plot(lambda_fit, double_gaussian_function(lambda_fit, *popt), '-', label='2 Gaussians fit', color=color_fitted_function)
            ax.legend()
            ax.set_title(title, fontsize=7)
            # plot data
            ax.plot(df_to_plot['xs'], df_to_plot['ys'], '.', alpha=1.0, color=color_data)
            ax.plot(df_to_plot['xs'], df_to_plot['ys'], '--', alpha=0.3, color=color_data)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.grid(True, alpha = 0.5)

        return fit_results

    def fit(self, lambda0_guess, delta_lambda=.5, sigma_guess = .01, gamma_guess = .01, background_type = 'linear', bounds = None, plot = False):
        """
        Fit the spectrum with a Voigt function

        :param lambda0_guess: (float) [nm]  guess value of the position of the Voigt function
        :param delta_lambda: (float) [nm]  range of wavelengths centered around lambda0_guess, used to fit the Voigt profile
        :param sigma_guess: (float) guess value of the sigma parameter of the Voigt function related to the linewidth of the Gaussian component
        :param gamma_guess: (float) guess value of the gamma parameter of the Voigt function related to the linewidth of the Lorenzian component
        :param background_type: (str) defines the type of background function to fit. Accepted values are 
           - "none"    : only Voigt profile is fitted
           - "const"   : a Voigt profile with constant value is fitted
           - "linear"  : a Voigt profile with linear background is fitted (y = slope * lambda + intercept)
        :param bounds: (dict) : dictionary containing the limit values that the function parameters can reach during the fitting procedure. [!! Warning : only implemented for background : "none"]. Available keys 
           - sigma_min 
           - sigma_max 
        :param plot: (bool) : If True, plot the spectrum and the fitted function

        :return: 
        - dict of result if successful
        - None if failed
        """

        # set the bounds if specified
        bound_sigma_min = 0
        bound_sigma_max = numpy.infty
        if bounds != None:
            if 'sigma_min' in bounds:
                bound_sigma_min = bounds['sigma_min']
            if 'sigma_max' in bounds:
                bound_sigma_max = bounds['sigma_max']


        try:
                # preparation of the data to fit :
                i0_guess = (numpy.abs(self.wavelengths - lambda0_guess)).argmin()
                i0_min =  (numpy.abs(self.wavelengths - (lambda0_guess - delta_lambda/2.))).argmin()
                i0_max =  (numpy.abs(self.wavelengths - (lambda0_guess + delta_lambda/2.))).argmin()
                # crop the spectrum to do the fit 
                xs = numpy.array(self.wavelengths[i0_min:i0_max]) # wavelengths (nm)
                ys = numpy.array(self.intensities[i0_min:i0_max]) # intensities (a.u.)

                if background_type=='linear': # fits a Voigt profile function + a linear function on the data
                    slope_guess = (ys[-1] - ys[0])/(xs[-1] - ys[0])
                    intercept_guess = ys[-1] - slope_guess*xs[-1]
                    a_guess = self.intensities[i0_guess]

                    p0=[lambda0_guess, a_guess, sigma_guess, gamma_guess, slope_guess, intercept_guess]

                    # non physical bounds to make the code using no bounds, still work
                    # set the bounds if specified
                    bound_sigma_min = -numpy.infty
                    bound_sigma_max = numpy.infty
                    if bounds != None:
                        if 'sigma_min' in bounds:
                            bound_sigma_min = bounds['sigma_min']
                        if 'sigma_max' in bounds:
                            bound_sigma_max = bounds['sigma_max']

                    bound_min = [-numpy.infty, -numpy.infty, bound_sigma_min, -numpy.infty, -numpy.infty, -numpy.infty]
                    bound_max = [numpy.infty, numpy.infty, bound_sigma_max, numpy.infty, numpy.infty, numpy.infty]

                    popt, pconv = curve_fit(voigt_profile_plus_linear_func, xs, ys, p0=p0, bounds=(bound_min, bound_max))
                    lambda0 = popt[0]
                    a = popt[1]
                    sigma = numpy.abs(popt[2])
                    gamma = numpy.abs(popt[3])
                    slope = popt[4]
                    intercept = popt[5]
                    f_g = 2*sigma*numpy.sqrt(2*numpy.log(2)) # FWHM of the gaussian part
                    f_l = 2*gamma # FWHM of the Lorentzian part
                    f_v = 0.5343*f_l + numpy.sqrt(0.2169*f_l**2 + f_g**2) # FWHM of a Voigt-profile (from https://en.wikipedia.org/wiki/Voigt_profile)
                    mu = f_l / (f_l + f_g) # shape factor (Aris) (unitless)
                    q = lambda0/f_v # Q-factor according to the Voigt profile linewidth
                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'voigt_profile_plus_linear_function',
                                        'lambda0_guess' : lambda0_guess,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'gamma_guess' : gamma_guess,
                                        'slope_guess'      : slope_guess,
                                        'intercept_guess'      : intercept_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0, # central wavelength (nm)
                                        'a' : a, # amplitude of the Voigt profile (a.u.)
                                        'sigma' : sigma, 
                                        'gamma' : gamma,
                                        'slope' : slope,
                                        'intercept' : intercept,
                                        'f_g' : f_g,
                                        'f_l' : f_l,
                                        'f_v' : f_v,
                                        'mu'  : mu,
                                        'q' : q,
                    }
                    self.fits.append(fit_result)

                    if plot:
                        ANNOTATION_FONT = 6
                        ANNOTATION_ALPHA = 0.5
                        fig, ax = plt.subplots()
                        # plot data
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        # plot fitted linear function taking into account the linear background
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001).tolist()
                        intensities_fit = linear_function(wavelength_fit, float(fit_result['slope']), float(fit_result['intercept']))
                        ax.plot(wavelength_fit, intensities_fit, color='mediumorchid', linestyle='--')
                        # plot the fitted function
                        intensities_fit = voigt_profile_plus_linear_func(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'], fit_result['gamma'], fit_result['slope'], fit_result['intercept'])
                        ax.plot(wavelength_fit, intensities_fit, label='Voigt fit (Q = '+"{:.2f}".format(fit_result['q'])+')', zorder=-2, color='mediumorchid')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result


                elif background_type=='const': # fits a Voigt profile function + a constant value
                    b_guess = max(self.intensities[i0_max], 0)
                    guess_intensity = self.intensities[i0_guess]
                    p0=[lambda0_guess, guess_intensity, sigma_guess, gamma_guess, b_guess]
                    popt, pconv = curve_fit(voigt_profile_func, xs, ys, p0=p0, bounds=([0, 0, 0, 0, 0], [numpy.infty,numpy.infty,numpy.infty,numpy.infty,numpy.infty]))
                    lambda0 = popt[0]
                    a = popt[1]
                    sigma = numpy.abs(popt[2])
                    gamma = numpy.abs(popt[3])
                    b = popt[4]
                    f_g = 2*sigma*numpy.sqrt(2*numpy.log(2)) # FWHM of the gaussian part
                    f_l = 2*gamma # FWHM of the Lorentzian part
                    f_v = 0.5346*f_l + numpy.sqrt(0.2166*f_l**2 + f_g**2) # FWHM of a Voigt-profile (from https://en.wikipedia.org/wiki/Voigt_profile)
                    q = lambda0/f_v
                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'voigt_profile',
                                        'lambda0_guess' : lambda0_guess,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'gamma_guess' : gamma_guess,
                                        'b_guess'      : b_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0,
                                        'a' : a,
                                        'sigma' : sigma,
                                        'gamma' : gamma,
                                        'b' : b,
                                        'f_g' : f_g,
                                        'f_l' : f_l,
                                        'f_v' : f_v,
                                        'q' : q,
                    }
                    self.fits.append(fit_result)

                    if plot:
                        ANNOTATION_FONT = 6
                        ANNOTATION_ALPHA = 0.5
                        fig, ax = plt.subplots()
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        # plot fitted constant background
                        ax.plot([min(xs), max(xs)], [b, b], color='mediumorchid', linestyle='--')
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001).tolist()
                        intensities_fit = voigt_profile_func(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'], fit_result['gamma'], fit_result['b'])
                        #ax.plot(wavelength_fit, intensities_fit, label='Voigt fit (Q = '+"{:.2f}".format(fit_result['q'])+')', zorder=-2, color='mediumorchid')
                        ax.plot(wavelength_fit, intensities_fit, label='Voigt fit (Q = '+"{:.2f}".format(fit_result['q'])+')', color='fuchsia')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result

                elif background_type=='none': # fits only a Voigt profile function 
                    guess_intensity = self.intensities[i0_guess]
                    p0=[lambda0_guess, guess_intensity, sigma_guess, gamma_guess]
                    bound_min = [0, 0, bound_sigma_min, 0]
                    bound_max = [numpy.infty, numpy.infty, bound_sigma_max, numpy.infty]
                    popt, pconv = curve_fit(voigt_profile_no_const, xs, ys, p0=p0, bounds=(bound_min, bound_max))
                    lambda0 = popt[0]
                    a = popt[1]
                    sigma = numpy.abs(popt[2])
                    gamma = numpy.abs(popt[3])
                    f_g = 2*sigma*numpy.sqrt(2*numpy.log(2)) # FWHM of the gaussian part
                    f_l = 2*gamma # FWHM of the Lorentzian part
                    f_v = 0.5346*f_l + numpy.sqrt(0.2166*f_l**2 + f_g**2) # FWHM of a Voigt-profile (from https://en.wikipedia.org/wiki/Voigt_profile)
                    q = lambda0/f_v
                    fit_result = {
                                        'x' : self.x,
                                        'fit_function' : 'voigt_profile',
                                        'lambda0_guess' : lambda0_guess,
                                        'delta_lambda' : delta_lambda,
                                        'sigma_guess' : sigma_guess,
                                        'gamma_guess' : gamma_guess,
                                        'x_min' : xs[0],
                                        'x_max' : xs[-1],
                                        'lambda0' : lambda0,
                                        'a' : a,
                                        'sigma' : sigma,
                                        'gamma' : gamma,
                                        'f_g' : f_g,
                                        'f_l' : f_l,
                                        'f_v' : f_v,
                                        'q' : q,
                    }
                    self.fits.append(fit_result)

                    if plot:
                        ANNOTATION_FONT = 6
                        ANNOTATION_ALPHA = 0.5
                        fig, ax = plt.subplots()
                        color='dodgerblue'
                        ax.plot(xs, ys, '.', alpha=1.0, color=color)
                        ax.plot(xs, ys, '--', alpha=0.3, color=color)
                        wavelength_fit = numpy.arange(min(xs), max(xs), 0.001).tolist()
                        intensities_fit = voigt_profile_no_const(wavelength_fit, fit_result['lambda0'], fit_result['a'], fit_result['sigma'], fit_result['gamma'])
                        ax.plot(wavelength_fit, intensities_fit, label='Voigt fit (Q = '+"{:.2f}".format(fit_result['q'])+')', color='fuchsia')
                        ax.set_xlabel('Wavelength (nm)')
                        ax.set_ylabel('Intensity (a.u.)')
                        ax.grid(True)
                        title = 'Fit Spectrum, background_type=\''+background_type+'\'\n' + os.path.basename(self.FILEPATH) + ', x=' + str(self.x)
                        ax.set_title(title, fontsize=7)

                    return fit_result

        except RuntimeError:
            print('x = ' + str(self.x) + ' : Voigt-profile fit failed : scipy.optimize.curve_fit() failed')
            return None

    

    def fit_2_voigts(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess, gamma_1_guess, x0_2_guess, a_2_guess, sigma_2_guess, gamma_2_guess, b_guess, bounds = None, plot=False, lambda_min_plot = None,  lambda_max_plot = None):
        """
        Fit 2 Voigt profiles on the spectrum

        :param lambda_min_fit: (float) [nm] minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] maximal wavelength used to fit the function

        for i being 1 and 2 : 

        :param x0_i_guess: (float, nm) guess value of the position of the i-th Voigt profile
        :param a_i_guess: (float) guess value of the amplitude of the i-th Voigt profile
        :param sigma_i_guess: (float) guess value of the sigma parameter (Gaussian STD) of the i-th Voigt profile
        :param gamma_i_guess: (float) guess value of the gamma parameter (Lorentzian FWHM) of the i-th Voigt profile

        :param b_guess: (float) constant value
        :param bounds: (dict) dictionary containing constraints of the parameters to be fitted. Available bounds are
            - "x0_1_min" (float)
            - "x0_1_max" (float)
            - "x0_2_min" (float)
            - "x0_2_max" (float)
            - "sigma_1_min" (float) 
            - "sigma_2_min" (float) 
            - "gamma_1_min" (float) 
            - "gamma_2_min" (float) 
        :param plot: (bool)  if yes, plots the data with the fitted function
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 

        :return: dictionary of result if successful, None if failed
        """

        # plot parameters
        color_data ='dodgerblue'
        ansatz_color = 'orange'
        voigt_profile_1_color = 'mediumseagreen'
        voigt_profile_2_color = 'crimson'
        alpha_individual_voigt_profile_ansatz = 0.7
        color_fitting_range = 'slategrey'
        alpha_fitting_range = 0.1
        color_fitted_function = 'fuchsia'

        if lambda_min_plot == None:
            lambda_min_plot = lambda_min_fit
        if lambda_max_plot == None:
            lambda_max_plot = lambda_max_fit
            
        df_to_fit = pandas.DataFrame({'xs':self.wavelengths, 'ys':self.intensities})
        df_to_fit = df_to_fit[(df_to_fit['xs'] >= lambda_min_fit) & (df_to_fit['xs'] <= lambda_max_fit)]
        normalization_factor = numpy.max(df_to_fit['ys'])
        df_to_fit['ys'] /= normalization_factor
        df_to_plot = pandas.DataFrame({'xs':self.wavelengths, 'ys':self.intensities})
        df_to_plot = df_to_plot[(df_to_plot['xs'] >= lambda_min_plot) & (df_to_plot['xs'] <= lambda_max_plot)]

        p0 = [  x0_1_guess, a_1_guess,  sigma_1_guess, gamma_1_guess,   # Voigt profile 1
                x0_2_guess, a_2_guess,  sigma_2_guess, gamma_2_guess,   # Voigt profile 2
                b_guess ]
        try : 

            a_1_guess_min = 1e-2 / normalization_factor
            a_2_guess_min = 1e-2 / normalization_factor
            bound_min = [lambda_min_fit, a_1_guess_min, 0, 0, 
                         lambda_min_fit, a_2_guess_min, 0, 0, 
                         0]
            a_1_guess_max = 2 * a_1_guess / normalization_factor
            a_2_guess_max = 2 * a_2_guess / normalization_factor
            bound_max = [lambda_max_fit, a_1_guess_max, 10, 10, 
                         lambda_max_fit, a_2_guess_max, 10, 10, 
                         numpy.infty]

            # apply bounds from the 'bounds' variables
            if bounds != None:
                # Voigt profile 1
                if 'x0_1_min' in bounds:
                    bound_min[0] = bounds['x0_1_min']
                if 'x0_1_max' in bounds:
                    bound_max[0] = bounds['x0_1_max']

                if 'sigma_1_min' in bounds:
                    bound_min[2] = bounds['sigma_1_min']
                if 'gamma_1_min' in bounds:
                    bound_min[3] = bounds['gamma_1_min']
                # Voigt profile 2
                if 'x0_2_min' in bounds:
                    bound_min[4] = bounds['x0_2_min']
                if 'x0_2_max' in bounds:
                    bound_max[4] = bounds['x0_2_max']
                if 'sigma_2_min' in bounds:
                    bound_min[6] = bounds['sigma_2_min']
                if 'gamma_2_min' in bounds:
                    bound_min[7] = bounds['gamma_2_min']
            p0[1] /= normalization_factor
            p0[5] /= normalization_factor
            p0[8] /= normalization_factor
            # fit 
            popt, pcov = curve_fit(double_voigt_profile_func, df_to_fit['xs'], df_to_fit['ys'], p0=p0, bounds=(bound_min, bound_max))
            popt[1] *= normalization_factor
            popt[5] *= normalization_factor
            popt[8] *= normalization_factor
            p0[1] *= normalization_factor
            p0[5] *= normalization_factor
            p0[8] *= normalization_factor

            # results Voigt 1
            x0_1 = popt[0]
            a_1 = popt[1] 
            sigma_1 = popt[2]
            gamma_1 = popt[3]
            f_g_1 = 2*sigma_1*numpy.sqrt(2*numpy.log(2)) # FWHM of the gaussian part
            f_l_1 = 2*gamma_1 # FWHM of the Lorentzian part
            f_v_1 = 0.5346*f_l_1 + numpy.sqrt(0.2166*f_l_1**2 + f_g_1**2) # FWHM of a Voigt-profile (from https://en.wikipedia.org/wiki/Voigt_profile)
            q_1 = x0_1 / f_v_1
            # results Voigt 2
            x0_2 = popt[4]
            a_2 = popt[5]
            sigma_2 = popt[6]
            gamma_2 = popt[7]
            f_g_2 = 2*sigma_2*numpy.sqrt(2*numpy.log(2)) # FWHM of the gaussian part
            f_l_2 = 2*gamma_2 # FWHM of the Lorentzian part
            f_v_2 = 0.5346*f_l_2 + numpy.sqrt(0.2166*f_l_2**2 + f_g_2**2) # FWHM of a Voigt-profile (from https://en.wikipedia.org/wiki/Voigt_profile)
            q_2 = x0_2 / f_v_2
            b = popt[8]

            # keep peak 1 to the left and peak 2 to the right
            if x0_1 > x0_2:
                x0_1, x0_2 =  x0_2, x0_1
                a_1, a_2 =  a_2, a_1
                sigma_1, sigma_2 =  sigma_2, sigma_1
                gamma_1, gamma_2 =  gamma_2, gamma_1
                f_g_1, f_g_2 =  f_g_2, f_g_1
                f_l_1, f_l_2 =  f_l_2, f_l_1
                f_v_1, f_v_2 =  f_v_2, f_v_1
                q_1, q_2 =  q_2, q_1

            fit_results = {
                                'x' : self.x,
                                'fit_function' : '2_voigt_profiles',
                                'lambda_min_fit' : lambda_min_fit,
                                'lambda_max_fit' : lambda_max_fit,
                                'x0_1_guess' : x0_1_guess,
                                'a_1_guess' : a_1_guess,
                                'sigma_1_guess' : sigma_1_guess,
                                'gamma_1_guess' : gamma_1_guess,
                                'x0_2_guess' : x0_2_guess,
                                'a_2_guess ' : a_2_guess,
                                'sigma_2_guess' : sigma_2_guess,
                                'gamma_2_guess' : gamma_2_guess,
                                'b_guess'       : b_guess,
                                'x0_1'      : x0_1,
                                'a_1'      : a_1,
                                'sigma_1'      : sigma_1,
                                'gamma_1'      : gamma_1,
                                'f_g_1'      : f_g_1,
                                'f_l_1'      : f_l_1,
                                'f_v_1'      : f_v_1,
                                'q_1'      : q_1,
                                'x0_2'      : x0_2,
                                'a_2'      : a_2,
                                'sigma_2'      : sigma_2,
                                'gamma_2'      : gamma_2,
                                'f_g_2'      : f_g_2,
                                'f_l_2'      : f_l_2,
                                'f_v_2'      : f_v_2,
                                'q_2'      : q_2,
                                'b'        : b
            }
            self.fits.append(fit_results)

        except RuntimeError:
            print('x = ' + str(self.x) + ' : Voigt-profile fit failed : scipy.optimize.curve_fit() failed')

            # plot data with ansatz function
            p0[1] *= normalization_factor
            p0[5] *= normalization_factor
            p0[8] *= normalization_factor

            if plot:
                title = 'Overview' + ',Fit Spectrum\n' + os.path.basename(self.FILEPATH) + ', x=' + "{:.4f}".format(self.x)
                fig, ax = plt.subplots()
                ax.set_title(title)
                # plot data
                df = pandas.DataFrame({'x':df_to_plot['xs'], 'y':df_to_plot['ys']})
                ax.plot(df_to_plot['xs'], df_to_plot['ys'], '.', alpha=1.0, color=color_data)
                ax.plot(df_to_plot['xs'], df_to_plot['ys'], '--', alpha=0.3, color=color_data)
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.grid(True, alpha = 0.5)
                ax.axvspan(xmin = lambda_min_fit, xmax = lambda_max_fit, color=color_fitting_range, alpha=alpha_fitting_range)
                lambda_fit = numpy.arange(lambda_min_plot, lambda_max_plot, (lambda_max_plot - lambda_min_plot)/1e3)
                # plot ansatz
                ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *p0[0:4], slope=0, intercept=p0[-1]), ':', label='Voigt 1 Ansatz', color=ansatz_color)
                ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *p0[4:8], slope=0, intercept=p0[-1]), ':', label='Voigt 2 Ansatz', color=ansatz_color)
                ansatz_function = double_voigt_profile_func(lambda_fit, *p0)
                ax.plot(lambda_fit, ansatz_function, '-', label='2-Voigts Ansatz', color='orange')

            return None
        

        if plot:
            
            # ----------- plots
            fig, ax = plt.subplots()
            title = 'Fit Spectrum\n' + os.path.basename(self.FILEPATH) + ', x=' + "{:.4f}".format(self.x)
            # plot data
            ax.set_title(title)
            lambda_fit = numpy.arange(lambda_min_plot, lambda_max_plot, (lambda_max_plot - lambda_min_plot)/1e3)
            # plot ansatz
            ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *p0[0:4], slope=0, intercept=b), ':', label='Voigt 1 Ansatz', color=voigt_profile_1_color, alpha = alpha_individual_voigt_profile_ansatz)
            ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *p0[4:8], slope=0, intercept=b), ':', label='Voigt 2 Ansatz', color=voigt_profile_2_color, alpha = alpha_individual_voigt_profile_ansatz)
            ansatz_function = double_voigt_profile_func(lambda_fit, *p0)
            ax.plot(lambda_fit, ansatz_function, '-', label='2-Voigts Ansatz', color=ansatz_color)

            # plot position of the fitted peaks
            ax.axvline(x0_1, linestyle='-', color=voigt_profile_1_color, alpha = 0.8)
            ax.axvline(x0_2, linestyle='-', color=voigt_profile_2_color, alpha = 0.8)

            # plot position peak bounds
            alpha_peak_bound_line = 1
            if bounds != None:
                # position Voigt profile 1
                if 'x0_1_min' in bounds:
                    ax.axvline(bounds['x0_1_min'], linestyle='-', color='black', alpha = alpha_peak_bound_line)
                if 'x0_1_max' in bounds:
                    ax.axvline(bounds['x0_1_max'], linestyle='--', color='black', alpha = alpha_peak_bound_line)
                # position Voigt profile 2
                if 'x0_2_min' in bounds:
                    ax.axvline(bounds['x0_2_min'], linestyle='-', color='black', alpha = alpha_peak_bound_line)
                if 'x0_2_max' in bounds:
                    ax.axvline(bounds['x0_2_max'], linestyle='--', color='black', alpha = alpha_peak_bound_line)

            # plot single Voigt profiles
            ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *popt[0:4], slope=0, intercept=b), label='Voigt 1', color=voigt_profile_1_color)
            ax.plot(lambda_fit, voigt_profile_plus_linear_func(lambda_fit, *popt[4:8], slope=0, intercept=b), label='Voigt 2', color=voigt_profile_2_color)
            ax.axvspan(xmin = lambda_min_fit, xmax = lambda_max_fit, color=color_fitting_range, alpha=alpha_fitting_range)

            # plot fitted function
            ax.plot(lambda_fit, double_voigt_profile_func(lambda_fit, *popt), '-', label='2-Voigts fit', color=color_fitted_function)
            ax.legend()
            ax.set_title(title, fontsize=7)

            # plot data
            ax.plot(df_to_plot['xs'], df_to_plot['ys'], '.', alpha=1.0, color=color_data)
            ax.plot(df_to_plot['xs'], df_to_plot['ys'], '--', alpha=0.3, color=color_data)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.grid(True, alpha = 0.5)

        return fit_results


    