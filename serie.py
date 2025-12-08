# Léo J. Roche (leoroche2@gmail.com)
# 04/12/2025

import pandas, numpy, os, json, re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cm
from nptdms import TdmsFile
from spectrum import Spectrum
from tools import *

class Serie:

    spectra: list[Spectrum]
    "List of `Spectrum` composing the `Serie`"

    xs : list[float]
    "List of the powers/currents corresponding to each `Spectrum`"

    FILEPATH : str
    "Filepath of the measurement file of the series"

    fits: list[dict]
    "List of dictionaries of the fits results "
    
    def __init__(self, FILEPATH):
        """
        Load `Serie` by reading a measurement file.
        Example : a power series measurement (one spectrum acquired with varied excitation power).
        The reading of new file formats can be implemented here (see below the example).
        Warning : The data file format has to be specified in the 'measurements_infos.json' file.

        :param FILEPATH: filepath of the measurement file to load

        """

        self.FILEPATH = FILEPATH
        self.spectra = []
        self.xs = []
        self.x_type = None # 'power' or 'current'
        self.fits = []

        # meas_infos_filepath = self.find_measurements_infos_file(FILEPATH)
        meas_infos_filepath = find_measurements_infos_file(FILEPATH)
        with open(meas_infos_filepath, 'r') as file:
            self.meas_infos = json.load(file)

        serie_fileformat = self.meas_infos['serie_fileformat']
        self.FILEPATH = FILEPATH


        # -------------------------------------------
        # Here add use cases to read new formats 
        # Example :
        # "
        # if serie_fileformat=='my_new_format':
        #   data_file = pandas.('my_txt_measurement_file.txt', 'rb')
        #   for power in data_file['power']:
        #       self.spectra.append(Spectrum(wavelengths=data_file.wavelengths, intensities=data_files.df[power].to_numpy(), x=power))
        # "
        # -------------------------------------------

        if (serie_fileformat=='qlab4_rt') or (serie_fileformat=='qlab2'):
            self.df = pandas.read_csv(FILEPATH, sep='\t')
            self.df = self.df.apply(pandas.to_numeric, errors='coerce', downcast='float')
            self.df = self.df.fillna(0)
            self.df = self.df.drop(['Energy'], axis=1)
            self.wavelengths = numpy.array(self.df['Wavelength'].tolist()[3:]) #  nm
            self.df.columns = self.df.loc[1]
            self.df = self.df.drop([0])
            self.df = self.df.drop([1])
            self.df = self.df.drop([2])
            # correct power
            POWER_FACTOR = float(self.meas_infos['x_factor'])
            new_powers = [x * POWER_FACTOR for x in self.df.columns.tolist()]
            self.df.columns = new_powers
            powers = self.df.columns.tolist()[1:]
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            # create Spectrum objects from DataFrame
            self.xs = powers
            self.x_type = 'power'
            self.spectra = []
            for power in powers:
                self.spectra.append(Spectrum(wavelengths=self.wavelengths, intensities=self.df[power].to_numpy(), x=power, FILEPATH=self.FILEPATH))

        elif serie_fileformat=='qlab4_rt_agilent': 
            # These scans can contain both I and V. By default x is associated with I, because most of the plots 
            # e.g. I/O S-curves are intensities in terms of I
            # Nonetheless, in 'Serie', V is also stored, in order to be able to plot the intensities in terms of V in 'plot_colormaps' (for Stark-Tuning plots)

            self.df = pandas.read_csv(FILEPATH, sep='\t')
            self.df = self.df.apply(pandas.to_numeric, errors='coerce', downcast='float')
            self.df = self.df.fillna(0)
            self.df = self.df.drop(['Energy'], axis=1)
            self.wavelengths = numpy.array(self.df['Wavelength'].tolist()[3:]) #  nm
            self.df.columns = self.df.loc[1]
            self.vs = self.df.loc[0].tolist()[1:] # V 
            self.df = self.df.drop([0])
            self.df = self.df.drop([1])
            self.df = self.df.drop([2])
            # correct power
            POWER_FACTOR = float(self.meas_infos['x_factor'])
            new_powers = [x * POWER_FACTOR for x in self.df.columns.tolist()]
            self.df.columns = new_powers
            powers = self.df.columns.tolist()[1:]
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            # create 'Spectrum' objects from DataFrame
            self.xs = powers
            self.x_type = 'power'
            self.spectra = []
            for power in powers:
                self.spectra.append(Spectrum(wavelengths=self.wavelengths, intensities=self.df[power].to_numpy(), x=power, FILEPATH=self.FILEPATH))

        elif serie_fileformat=='qlab4_rt_agilent_pola': 
            self.df = pandas.read_csv(FILEPATH, sep='\t')
            self.df = self.df.apply(pandas.to_numeric, errors='coerce', downcast='float')
            self.df = self.df.fillna(0)
            self.df = self.df.drop(['Energy'], axis=1)
            self.wavelengths = numpy.array(self.df['Wavelength'].tolist()[3:]) #  nm
            self.df.columns = self.df.loc[1]
            angles = self.df.loc[1].tolist()[1:] # deg
            self.df = self.df.drop([0])
            self.df = self.df.drop([1])
            self.df = self.df.drop([2])
            # create 'Spectrum' objects from DataFrame
            xs = angles
            x_type = 'angles'
            self.spectra = []
            for angle in angles:
                self.spectra.append(Spectrum(wavelengths=self.wavelengths, intensities=self.df[angle].to_numpy(), x=angle, FILEPATH=FILEPATH))


        elif serie_fileformat=='qlab5_picoprobe': 
            # checking if the ending of the filename is .tdms (current injection series) or .txt (polarization series)
            if FILEPATH[-5:]=='.tdms': # corresponds to a current series
                self.xs = []
                self.x_type = 'current' # injected current
                self.spectra = []
                tdms_file = TdmsFile.read(self.FILEPATH)
                n_frames = len(tdms_file["Spectrum: Intensity[4-4]"]['Frame'].data)
                frame_len = int(len(tdms_file["Spectrum: SpecAxis[4-5]"]['Data'].data) / n_frames)
                print('n_frames : '+str(n_frames))
                for frame in range(n_frames):
                    i_min, i_max = frame*frame_len, (frame+1)*frame_len
                    wavelengths = tdms_file['Spectrum: SpecAxis[4-5]']['Data'].data[i_min:i_max] # nm
                    intensities = tdms_file["Spectrum: Intensity[4-4]"]['Data'].data[i_min:i_max] # (a.u.)
                    current = tdms_file['IV-Single: current[7-4]']['Data'].data[frame] * 1e6 # µA
                    self.xs.append(current)
                    self.spectra.append(Spectrum(wavelengths=wavelengths, intensities=intensities, x=current, FILEPATH=self.FILEPATH))
            
            elif FILEPATH[-4:]=='.txt': # corresponds to a polarization series
                print('opening polarization serie...')
                self.xs = []
                self.x_type = 'angle' # polarization angle
                self.spectra = []
                df = pandas.read_csv(FILEPATH, sep='\t')
                df = df.drop(index=[0])
                df = df.astype(float)
                wavelengths = df.iloc[1:]['Wavelength'].values
                for column in df.columns:
                    if column != 'Wavelength':
                        angle = float(numpy.max(df.iloc[0][column]))
                        intensities = df.iloc[1:][column]
                        self.xs.append(angle)
                        self.spectra.append(Spectrum(wavelengths=wavelengths, intensities=intensities, x = angle, FILEPATH = FILEPATH))

                self.xs.reverse()
                self.spectra.reverse()
                
        else :
            print('serie_fileformat in \'measurement_infos.json\' not recognized')
            exit()


    def apply_OD(self):
        """
        Extract OD from filename and calculate with its value what is the intensity before the OD filter
        :return: self
        """
        match = re.search(r'_OD_([+-]?([0-9]*[.])?[0-9]+)__', self.FILEPATH)
        if match==None:
            self.OD = 0
        else : 
            self.OD = float(match.group(1))

        self.scale_intensity(10**(self.OD))
        return self

    def remove(self, lambda_min = None, lambda_max = None):
        """
        Remove from the spectrum the region [**lambda_min**, **lambda_max**] of all spectra of the `Serie`

        :param lambda_min: (float) [nm] minimal wavelength of the spectrum region to remove
        :param lambda_max: (float) [nm] maximal wavelength of the spectrum region to remove
        :return: self
        """
        for i in range(len(self.spectra)):
                self.spectra[i].remove(lambda_min=lambda_min, lambda_max=lambda_max)
        
        return self


    def scale_intensity(self, factor):
            """ 
            Multiply the intensity values of each spectra by a factor (can be used to stitch different `Series` manually)

            :param factor: (float) multiplying factor
            :return: self
            """
            for i in range(len(self.spectra)):
                    self.spectra[i].scale_intensity(factor)
            return self

    def scale_xs(self, factor):
            """ 
            Multiply the x of each spectra by **factor** 
            :param factor: (float) multiplying factor
            """
            for i in range(len(self.xs)):
                    self.xs[i] *= factor
                    self.spectra[i].x *= factor
            return self

    def get_max_x(self):
        """
        Return the maximum x value
        """
        return max(self.xs)

    def fit_gauss(self, lambda0_guess, x_min = None, x_max = None, delta_lambda=.5, sigma_guess = .01, bounds = None, background_type = 'linear', plot=False):
        """
        Fits all spectra of the `Serie` with a Gaussian function (for all spectra within [x_min, x_max])

        :param lambda0_guess: (float) [nm] guess value of the position of the Voigt function
        :param x_min: (float) [nm] minimum x values of the spectrum to be fitted
        :param x_max: (float) [nm] maximum x values of the spectrum to be fitted
        :param delta_lambda: (float) [nm] range of wavelengths centered around lambda0_guess, used to fit the Voigt profile
        :param sigma_guess: (float) [nm] guess value of the standard deviation of the Gaussian function
        :param bounds: (dict) [nm] dictionary containing the limit values that the function parameters can reach during the fitting procedure. Available keys are
            - "sigma_min"
            - "sigma_max"
        :param background_type: (str) defines the type of background function to fit. Accepted values are 
            - "none"    : Gaussian function only
            - "const"   : Gaussian function with constant 
            - "linear"  : Gaussian function with linear function (y = slope * lambda + intercept)
        :param plot: (bool)  if True, plots the spectra and the fitted function
        :return: Dictionary of fit result or None if the fit failed
        """

        if x_min == None:
            x_min = - numpy.infty
        if x_max == None:
            x_max = numpy.infty
        first_fit = True
        for i in range(len(self.spectra)): # starting with bigger x-values (high powers)
            if (self.spectra[i].x > x_min) and (self.spectra[i].x < x_max):
                if first_fit: # if first fit within this function, use the guess values provided when calling this function
                    fit_result = self.spectra[i].fit_gauss(
                            lambda0_guess           = lambda0_guess, 
                            delta_lambda            = delta_lambda, 
                            sigma_guess             = sigma_guess,
                            background_type       = background_type,
                            bounds                  = bounds,
                            plot                    = plot
                    )
                else: # use results from last fit in the same Serie as guess values
                    fit_result = self.spectra[i].fit_gauss(
                            lambda0_guess           = self.fits[-1]['lambda0'], 
                            delta_lambda           = delta_lambda, 
                            sigma_guess            = self.fits[-1]['sigma'],
                            background_type       = background_type,
                            bounds                  = bounds,
                            plot                    = plot
                    )

                if fit_result==None:
                    print('Serie.fit_gauss() : i = '+str(i) + 'could not be fitted')
                else: 
                    self.fits.append(fit_result)
                    first_fit = False


    def fit_2_gauss(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess,  x0_2_guess, a_2_guess, sigma_2_guess, b_guess, x_min = None, x_max = None, bounds = None,  plot=False, freeze_guesses = True, lambda_min_plot = None,  lambda_max_plot = None):
        """Fits all spectra of the `Serie` with 2 Gaussian functions (for all spectra within [x_min, x_max])

        :param lambda_min_fit: (float) [nm] minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] maximal wavelength used to fit the function
        :param x0_i_guess: (float) [nm] guess value of the position of the i-th Gaussian function
        :param a_i_guess: (float) guess value of the amplitude of the i-th Gaussian function
        :param sigma_i_guess: (float) guess value of the sigma parameter of the i-th Gaussian function
        :param b_guess: (float) constant value guess
        :param bounds: (dict) dictionary containing the minimum and maximum values the parameters to be fitted. Available bounds are
            - "x0_1_min" (float) 
            - "x0_2_min" (float) 
            - "x0_1_max" (float) 
            - "x0_2_max" (float) 
            - "sigma_1_min" (float) 
            - "sigma_2_min" (float) 
        :param plot: (bool) if true, plot the data with the fitted function
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 
        :return: dictionary of fit result or None if the fit failed
        """
        if x_min == None:
            x_min = - numpy.infty
        if x_max == None:
            x_max = numpy.infty

        first_fit = True
        for i in range(len(self.spectra)-1, -1, -1): # iterate backward, i.e. with increasing x value
            if (self.spectra[i].x > x_min) and (self.spectra[i].x < x_max):
                if first_fit or freeze_guesses: # if first fit within this function or freeze_guesses is True, use the guess values provided when calling this function
                    fit_result = self.spectra[i].fit_2_gauss(
                            lambda_min_fit      = lambda_min_fit, 
                            lambda_max_fit      = lambda_max_fit, 
                            x0_1_guess           = x0_1_guess, 
                            a_1_guess           = a_1_guess, 
                            sigma_1_guess           = sigma_1_guess, 
                            x0_2_guess           = x0_2_guess, 
                            a_2_guess           = a_2_guess, 
                            sigma_2_guess           = sigma_2_guess, 
                            b_guess                 = b_guess,
                            bounds                  = bounds,
                            plot                    = plot,
                            lambda_min_plot         = lambda_min_plot,
                            lambda_max_plot         = lambda_max_plot
                    )
                else: # use results from last fit in the same Serie as guess values
                    fit_result = self.spectra[i].fit_2_gauss(
                            lambda_min_fit      = lambda_min_fit, 
                            lambda_max_fit      = lambda_max_fit, 
                            x0_1_guess           = self.fits[-1]['x0_1'], 
                            a_1_guess           = self.fits[-1]['a_1'], 
                            sigma_1_guess           = self.fits[-1]['sigma_1'], 
                            x0_2_guess           = self.fits[-1]['x0_2'], 
                            a_2_guess           = self.fits[-1]['a_2'], 
                            sigma_2_guess           = self.fits[-1]['sigma_2'], 
                            b_guess           = self.fits[-1]['b'], 
                            bounds          = bounds,
                            plot                    = plot,
                            lambda_min_plot         = lambda_min_plot,
                            lambda_max_plot         = lambda_max_plot
                    )

                if fit_result==None:
                    print('Serie.fit_2_voigts() : i = '+str(i) + ' could not be fitted')
                else: 
                    self.fits.append(fit_result)
                    first_fit = False
       

    def fit(self, lambda0_guess, x_min = None, x_max = None, delta_lambda=.5, sigma_guess = .01, gamma_guess = .01, background_type = 'linear', bounds = None, plot=False):
        """Fits all spectra of the `Serie` with a Voigt profile (for all spectra within [x_min, x_max])

        :param lambda0_guess: (float) [nm] guess value of the position of the Voigt function
        :param x_min: (float) minimum x values of the spectrum to be fitted
        :param x_max: (float) maximum x values of the spectrum to be fitted
        :param delta_lambda: (float) [nm] range of wavelengths centered around lambda0_guess, used to fit the Voigt profile
        :param sigma_guess: (float) guess value of the sigma parameter of the Voigt function
        :param gamma_guess: (float) guess value of the gamma parameter of the Voigt function
        :param background_type: (str) defines the type of background function to fit. Accepted values are 
            - "none"    : Voigt profile only
            - "const"   : Voigt profile with b constant
            - "linear"  : Voigt profile with linear function for background (y = slope * lambda + intercept)
        :param bounds: (dict) dictionary containing the limit values that the function parameters can reach during the fitting procedure. [ONLY IMPLEMENTED FOR BACKGROUND "none"]. Available keys 
            - "sigma_min"
            - "sigma_max"
        :param plot: (bool) if True, plots the spectra and the fitted functions
        :return: Dictionary of fit result or None if the fit failed
        """

        if x_min == None:
            x_min = - numpy.infty
        if x_max == None:
            x_max = numpy.infty

        first_fit = True

        for i in range(len(self.spectra)): # starting with bigger x-values (high powers)
            if (self.spectra[i].x > x_min) and (self.spectra[i].x < x_max):
                if first_fit: # if first fit within this function, use the guess values provided when calling this function
                    fit_result = self.spectra[i].fit(
                            lambda0_guess           = lambda0_guess, 
                            delta_lambda            = delta_lambda, 
                            sigma_guess             = sigma_guess,
                            gamma_guess             = gamma_guess,
                            background_type       = background_type,
                            bounds                  = bounds, 
                            plot                    = plot
                    )
                else: # use results from last fit 
                    fit_result = self.spectra[i].fit(
                            lambda0_guess           = self.fits[-1]['lambda0'], 
                            delta_lambda           = delta_lambda, 
                            sigma_guess            = self.fits[-1]['sigma'],
                            gamma_guess            = self.fits[-1]['gamma'],
                            background_type       = background_type,
                            bounds                  = bounds, 
                            plot                    = plot
                    )

                if fit_result==None:
                    print('Serie.fit() : i = '+str(i) + 'couldn not be fitted')
                else: 
                    self.fits.append(fit_result)
                    first_fit = False

    
    def fit_2_voigts(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess, gamma_1_guess, x0_2_guess, a_2_guess, sigma_2_guess, gamma_2_guess, b_guess, bounds = None, lambda_min_plot = None, lambda_max_plot = None, x_min = None, x_max = None, freeze_guesses = False, plot=False):
        """
        Fit 2 Voigt profiles on the spectra of the `Serie` (ascending x values)

        :param lambda_min_fit: (float) [nm] minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] maximal wavelength used to fit the function
        :param x0_i_guess: (float) [nm] guess value of the position of the i-th Voigt profile
        :param a_i_guess: (float) [nm] guess value of the amplitude of the i-th Voigt profile
        :param sigma_i_guess: (float) guess value of the sigma parameter of the i-th Voigt profile
        :param gamma_i_guess: (float) guess value of the gamma parameter of the i-th Voigt profile
        :param b_guess: (float) constant guess value 
        :param bounds: (dict) dictionary containing the minimum and maximum values of the parameters to be fitted. Available bounds 
            - "x0_1_min" (float)
            - "x0_1_max" (float)
            - "x0_2_min" (float)
            - "x0_2_max" (float)
            - "sigma_1_min" (float) 
            - "sigma_2_min" (float) 
            - "gamma_1_min" (float) 
            - "gamma_2_min" (float) 
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 
        :param x_min: (float) minimal value of x to be fitted
        :param x_max: (float) maximal value of x to be fitted
        :param freeze_guesses: (bool) if True, the same guess values are use for all the fitting
        :param plot: (bool) if True, plot the data with the fitted function
        """
        if x_min == None:
            x_min = - numpy.infty
        if x_max == None:
            x_max = numpy.infty

        first_fit = True
        for i in range(len(self.spectra)-1, -1, -1): # iterate backward, i.e. with increasing x value
            if (self.spectra[i].x > x_min) and (self.spectra[i].x < x_max):
                if first_fit or freeze_guesses: # if first fit within this function or freeze_guesses is True, use the guess values provided when calling this function
                    fit_result = self.spectra[i].fit_2_voigts(
                            lambda_min_fit      = lambda_min_fit, 
                            lambda_max_fit      = lambda_max_fit, 
                            x0_1_guess           = x0_1_guess, 
                            a_1_guess           = a_1_guess, 
                            sigma_1_guess           = sigma_1_guess, 
                            gamma_1_guess           = gamma_1_guess, 
                            x0_2_guess           = x0_2_guess, 
                            a_2_guess           = a_2_guess, 
                            sigma_2_guess           = sigma_2_guess, 
                            gamma_2_guess           = gamma_2_guess, 
                            b_guess                 = b_guess,
                            bounds                  = bounds,
                            plot                    = plot,
                            lambda_min_plot         = lambda_min_plot,
                            lambda_max_plot         = lambda_max_plot
                    )
                else: # use results from last fit in the same Serie as guess values
                    fit_result = self.spectra[i].fit_2_voigts(
                            lambda_min_fit      = lambda_min_fit, 
                            lambda_max_fit      = lambda_max_fit, 
                            x0_1_guess           = self.fits[-1]['x0_1'], 
                            a_1_guess           = self.fits[-1]['a_1'], 
                            sigma_1_guess           = self.fits[-1]['sigma_1'], 
                            gamma_1_guess           = self.fits[-1]['gamma_1'], 
                            x0_2_guess           = self.fits[-1]['x0_2'], 
                            a_2_guess           = self.fits[-1]['a_2'], 
                            sigma_2_guess           = self.fits[-1]['sigma_2'], 
                            gamma_2_guess           = self.fits[-1]['gamma_2'], 
                            b_guess           = self.fits[-1]['b'], 
                            bounds          = bounds,
                            plot                    = plot,
                            lambda_min_plot         = lambda_min_plot,
                            lambda_max_plot         = lambda_max_plot
                    )

                if fit_result==None:
                    print('Serie.fit_2_voigts() : i = '+str(i) + ' could not be fitted')
                else: 
                    self.fits.append(fit_result)
                    first_fit = False


    def get_fit_results(self):
        """ 
        Return a pandas DataFrame containing the gathered fit results
        """
        df = pandas.DataFrame({})
        for fit in self.fits:
            df = df._append(fit, ignore_index=True)
        return df

    def plot(self, lambda_min = -numpy.infty, lambda_max = numpy.infty, show=False):
        """
        Plot all spectra on the same figure

        :param lambda_min: (float) [nm] minimum wavelength to be plotted
        :param lambda_max: (float) [nm] maximum wavelength to be plotted
        :param show: (bool) : if true, show the generated plot
        """
        plt.figure()
        colors = cm.summer(numpy.linspace(0, 1, len(self.spectra)))
        for i in range(len(self.spectra)):
            d = pandas.DataFrame({'wavelengths':self.spectra[i].wavelengths, 'intensities':self.spectra[i].intensities})
            d = d[(d['wavelengths']>=lambda_min) & (d['wavelengths']<=lambda_max)]
            plt.plot(d['wavelengths'], d['intensities'], '.', alpha=1.0, color=colors[i])
            plt.plot(d['wavelengths'], d['intensities'], '--', alpha=0.3, color=colors[i])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.grid(True)
        plt.title(os.path.basename(self.FILEPATH), fontsize=8)
        if show:
            plt.show()
        

    def plot_colormap(self, v_min=None, v_max=None, lambda_min=None, lambda_max=None, scale='lin'):
        """
        Plot colormap (useful for plotting Stark-Tuning measurements)

        :param v_min: (float) [V] minimum voltage value to be shown
        :param v_max: (float) [V] maximum voltage value to be shown
        :param lambda_min: (float) [nm] minimum wavelength value to be shown
        :param lambda_max: (float) [nm] maximum wavelength value to be shown
        :param scale: (str) scale of the color gradient used to show the intensity value (either "lin" or "log")
        """
        if v_min == None:
            v_min = -numpy.infty
        if v_max == None:
            v_max = numpy.infty
        df = pandas.DataFrame(index=self.spectra[0].wavelengths)
        for i in range(len(self.spectra)):
            df[self.vs[i]] = self.spectra[i].intensities

        df = df.loc[lambda_min:lambda_max]
        df = df.loc[:, [col for col in df.columns if col >= v_min]]
        df = df.loc[:, [col for col in df.columns if col <= v_max]]

        voltages = df.columns.tolist()
        wavelengths = df.index.values.tolist()
        X, Y = numpy.meshgrid(voltages, wavelengths)

        plt.figure()
        cmap = 'afmhot'
        if scale=='lin':
            plt.pcolormesh(X, Y, df.values, cmap=cmap, linewidth=0)
        elif scale=='log':
            min_value = 10**(numpy.floor(numpy.log10(numpy.min(df.values))))
            max_value = 10**(numpy.ceil(numpy.log10(numpy.max(df.values))))
            plt.pcolormesh(X, Y, df.values, cmap=cmap, linewidth=0, norm=colors.LogNorm(vmin=min_value, vmax=max_value))
        plt.xlabel('Applied Bias (V)')
        plt.ylabel('Wavelength (nm)')
        cbar = plt.colorbar()
        cbar.set_label('Intensity (a.u.)')

