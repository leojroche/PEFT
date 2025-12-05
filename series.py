# Léo J. Roche (leoroche2@gmail.com)
# 03/12/2025

import numpy, os, pandas
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from serie import Serie

class Series:

    FILEPATHS : str
    "Filepaths of the respective `Serie`s contained in the object"

    series : list[Serie]
    "List of `Serie`s"

    def __init__(self, FILEPATHS, stitch=True):
            self.FILEPATHS = FILEPATHS
            self.series = []
            self.fits = []
            for filepath in self.FILEPATHS:
                self.series.append(Serie(filepath))
            self.x_type = self.series[0].x_type
            self.title = 'Series\n'
            for filepath in self.FILEPATHS:
                self.title += os.path.basename(filepath) + '\n'
            # get max x from each serie for later stitching
            max_xs = []
            for serie in self.series:
                    max_xs.append(serie.get_max_x())   
            new_series = []
            sorted_indices = sorted(range(len(max_xs)), key=lambda i: max_xs[i], reverse=False)
            for i in range(len(sorted_indices)):
                new_series.append(self.series[sorted_indices[i]])

            self.series = new_series
            self.series.reverse()
            if stitch:
                self.stitch()
            else:
                self.apply_OD()


    def plot(self, lambda_min = -numpy.infty, lambda_max = numpy.infty, show=False, save=False):
        """
        Plot all spectra on the same figure

        :param lambda_min: (float) [nm] minimum wavelength to be plotted
        :param lambda_max: (float) [nm] maximum wavelength to be plotted
        :param show: (bool)  if true, show the generated plot
        :param save: (bool)  if true, save the generated plots
        :return: self
        """
        plt.figure()
        # create color gradient
        nb_spectra = 0
        for j in range(len(self.series)):
             nb_spectra += len(self.series[j].spectra)
        colors = cm.summer(numpy.linspace(0, 1, nb_spectra))

        k = 0
        for j in range(len(self.series)):
            for i in range(len(self.series[j].spectra)):
                d = pandas.DataFrame({'wavelengths':self.series[j].spectra[i].wavelengths, 'intensities':self.series[j].spectra[i].intensities})
                d = d[(d['wavelengths']>=lambda_min) & (d['wavelengths']<=lambda_max)]
                plt.plot(d['wavelengths'], d['intensities'], '.', alpha=1.0, color=colors[k])
                plt.plot(d['wavelengths'], d['intensities'], '--', alpha=0.3, color=colors[k])
                k += 1
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (a.u.)')
        plt.grid(True)
        plt.title(self.title, fontsize=8)
        if show:
            plt.show()
        if save:
            print('TODO : save the plot')
        return self

    def scale_xs(self, factor):
        """
        Scale the values of x by a multiplying factor

        :param: factor: (float) multiplying factor
        :return: self
        """

        for i in range(len(self.series)):
            self.series[i].scale_xs(factor)
        return self

    def apply_OD(self):
        """
        For each `Serie`, change the intensities to get the number of counts before the OD (alternative to stitching)

        :return: self
        """
        for serie in self.series:
            serie.apply_OD()
        return self

    def remove(self, lambda_min = None, lambda_max = None):
        """
        Remove the spectral the region [lambda_min, lambda_max].

        :param lambda_min: (float) [nm] minimal wavelength of the spectral region to be removed
        :param lambda_max: (float) [nm] maximal wavelength of the spectral region to be removed
        """
        for serie in self.series:
            serie.remove(lambda_min=lambda_min, lambda_max=lambda_max)

        return self

    def stitch(self) :  
        """
        Stitch the series using the maximum value
        :return: self 
        """

        if len(self.series)<=1:
            return self

        # get max x from each serie for later stitching
        max_xs = []
        for serie in self.series:
                max_xs.append(serie.get_max_x())   

        new_series = []
        # sort the 'new_series' in a new list with ascending order of respective maximum 'x'
        sorted_indices = sorted(range(len(max_xs)), key=lambda i: max_xs[i], reverse=False)
        for i in range(len(sorted_indices)):
            new_series.append(self.series[sorted_indices[i]])

        self.series = new_series

        # apply factors
        index = 0
        for i in range(1, len(self.series)):
            # stitching a (i) to b (i-1)
            max_x_b = max(self.series[i-1].xs)
            index_b = self.series[i-1].xs.index(max_x_b)
            index_a = (numpy.abs(numpy.array(self.series[i].xs) - max_x_b)).argmin()
            x_a = self.series[i].xs[index_a]
            a = self.series[i].spectra[index_a].intensities[index]
            b = self.series[i-1].spectra[index_b].intensities[index]
            self.series[i].scale_intensity(b/a)
        self.series.reverse()

        # update FILEPATHS
        self.FILEPATHS = []
        for i in range(len(self.series)):
            self.FILEPATHS.append(self.series[i].FILEPATH)
        # update title
        self.title = ''
        for filepath in self.FILEPATHS:
            self.title += os.path.basename(filepath) + '\n'

        # plot for debuggin : 
        print('checking stitching...')
        for i in range(len(self.series)):
            print('i')
            print(i)
            print('self.series[i].xs')
            print(self.series[i].xs)


    def verify_stitching(self, index):
        """
        Verify the stitching by plotting the intensity at a specific wavelength defined by the **index** in terms of **x**.

        :param index: (int) index of the intensities to be plotted
        """
        
        plt.figure()
        for i in range(len(self.series)):
            xs = []
            ys = []
            for spectrum in self.series[i].spectra:
                xs.append(spectrum.x)
                ys.append(spectrum.intensities[index])
            plt.plot(xs, ys, '.', label=str(i))
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('Intensity['+str(index)+']')

    def fit(self, lambda0_guess, x_min = None, x_max = None, delta_lambda=.5, sigma_guess = .01, gamma_guess = .01, background_type = 'linear', bounds = None, plot=False):
        """
        Fit a Voigt profile on the spectra.

        :param lambda0_guess: (float) [nm] guess value of the position of the Voigt function
        :param x_min: (float) minimum x values to be fitted
        :param x_max: (float) maximum x values to be fitted
        :param delta_lambda: (float) [nm] range of wavelengths centered around lambda0_guess, used to fit the Voigt profile
        :param sigma_guess: (float) guess value of the sigma parameter of the Voigt function
        :param gamma_guess: (float) guess value of the gamma parameter of the Voigt function
        :param background_type: (string) : 
            - "none"    : Voigt profile only
            - "const"   : Voigt profile with constant 
            - "linear"  : Voigt profile with linear function 
        :param bounds: (dict) dictionary containing the limit values that the function parameters can reach during the fitting procedure. [ONLY IMPLEMENTED FOR BACKGROUND 'none'] Available keys are
            - "sigma_min"
            - "sigma_max"
        :param plot: (bool) If true, plots the spectra and the fitted functions

        :return: Dictionary of fit result or None if the fit failed
        """

        if x_min ==  None:
            x_min = - numpy.infty
        if x_max ==  None:
            x_max = numpy.infty

        first_fit = True
        self.fits.append([])
        for i in range(len(self.series)):
            for j in range(len(self.series[i].spectra)): # starting with bigger x-values (high powers)
                if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                    if first_fit:
                        fit_result = self.series[i].spectra[j].fit(
                                lambda0_guess           = lambda0_guess, 
                                delta_lambda            = delta_lambda, 
                                sigma_guess             = sigma_guess,
                                gamma_guess             = gamma_guess,
                                background_type       = background_type,
                                bounds                  = bounds,
                                plot                    = plot
                        )
                    else: # use results from last fit 
                        fit_result = self.series[i].spectra[j].fit(
                                lambda0_guess           = self.fits[-1][-1]['lambda0'], 
                                delta_lambda            = delta_lambda, 
                                sigma_guess             = self.fits[-1][-1]['sigma'],
                                gamma_guess             = self.fits[-1][-1]['gamma'],
                                background_type       = background_type,
                                bounds                  = bounds,
                                plot                    = plot
                        )
                    if fit_result==None:
                        print('(i, j) = ('+str(i)+', '+str(j)+')')
                    else: 
                        new_fit_result = {'index_series':i, 'index_spectra':j}
                        new_fit_result.update(fit_result)
                        self.fits[-1].append(new_fit_result)
                        first_fit = False


    def fit_2_voigts(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess, gamma_1_guess, x0_2_guess, a_2_guess, sigma_2_guess, gamma_2_guess, b_guess, bounds = None, x_min = None, x_max = None, freeze_guesses = False, plot=False, descending=False, lambda_min_plot = None, lambda_max_plot = None):
        """
        Fits the spectra with a double Voigt profile 

        :param lambda_min_fit: (float) [nm] minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] maximal wavelength used to fit the function
        :param x0_i_guess: (float) [nm] guess value of the position of the i-th Voigt profile
        :param a_i_guess: (float) guess value of the amplitude of the i-th Voigt profile
        :param sigma_i_guess: (float) guess value of the sigma parameter of the i-th Voigt profile
        :param gamma_i_guess: (float) guess value of the gamma parameter of the i-th Voigt profile
        :param b_guess: (float) constant value
        :param bounds: (dict) dictionary containing the minimum and maximum values of the parameters to be fitted. Available bounds are
            - "x0_1_min" (float)
            - "x0_1_max" (float)
            - "x0_2_min" (float)
            - "x0_2_max" (float)
            - "sigma_1_min" (float) 
            - "sigma_2_min" (float) 
            - "gamma_1_min" (float) 
            - "gamma_2_min" (float) 
        :param x_min: (float) minimal value of x to be fitted
        :param x_max: (float) maximal value of x to be fitted
        :param freeze_guesses: (bool) if true, the same guess values are used for all the fitting
        :param plot: (bool) if true, plot the data with the fitted function
        :param descending: (bool) if true, fit the spectra by decreasing order of their respective x value
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 

        :return: dictionary of fit result or None if failed
        """

        if x_min ==  None:
            x_min = - numpy.infty
        if x_max ==  None:
            x_max = numpy.infty

        first_fit = True
        self.fits.append([])
        if descending==False: # iterate with x increasing 
            for i in range(len(self.series)-1, -1, -1): 
                for j in range(len(self.series[i].spectra)-1, -1, -1): 
                    if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                        if first_fit or freeze_guesses:
                            fit_result = self.series[i].spectra[j].fit_2_voigts(
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
                                b_guess                       = b_guess,
                                bounds          = bounds,
                                plot                    = plot,
                                lambda_min_plot         = lambda_min_plot,
                                lambda_max_plot         = lambda_max_plot
                            )
                        else: # use results from last fit 
                            fit_result = self.series[i].spectra[j].fit_2_voigts(
                                        lambda_min_fit      = lambda_min_fit, 
                                        lambda_max_fit      = lambda_max_fit, 
                                        x0_1_guess           = self.fits[-1][-1]['x0_1'], 
                                        a_1_guess           = self.fits[-1][-1]['a_1'], 
                                        sigma_1_guess           = self.fits[-1][-1]['sigma_1'], 
                                        gamma_1_guess           = self.fits[-1][-1]['gamma_1'], 
                                        x0_2_guess           = self.fits[-1][-1]['x0_2'], 
                                        a_2_guess           = self.fits[-1][-1]['a_2'], 
                                        sigma_2_guess           = self.fits[-1][-1]['sigma_2'], 
                                        gamma_2_guess           = self.fits[-1][-1]['gamma_2'], 
                                        b_guess                       = self.fits[-1][-1]['b'], 
                                        bounds          = bounds,
                                        plot                    = plot,
                                        lambda_min_plot         = lambda_min_plot,
                                        lambda_max_plot         = lambda_max_plot
                            )

                        if fit_result==None:
                            print('(i, j) = ('+str(i)+', '+str(j)+')')
                        else: 
                            new_fit_result = {'index_series':i, 'index_spectra':j}
                            new_fit_result.update(fit_result)
                            self.fits[-1].append(new_fit_result)
                            first_fit = False
        else: 
            for i in range(len(self.series)):
                for j in range(len(self.series[i].spectra)): # starting with bigger x-values (high powers)
                    if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                        if first_fit or freeze_guesses:
                            fit_result = self.series[i].spectra[j].fit_2_voigts(
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
                                b_guess                       = b_guess,
                                bounds          = bounds,
                                plot                    = plot,
                                lambda_min_plot         = lambda_min_plot,
                                lambda_max_plot         = lambda_max_plot
                            )
                        else: # use results from last fit 
                            fit_result = self.series[i].spectra[j].fit_2_voigts(
                                        lambda_min_fit      = lambda_min_fit, 
                                        lambda_max_fit      = lambda_max_fit, 
                                        x0_1_guess           = self.fits[-1][-1]['x0_1'], 
                                        a_1_guess           = self.fits[-1][-1]['a_1'], 
                                        sigma_1_guess           = self.fits[-1][-1]['sigma_1'], 
                                        gamma_1_guess           = self.fits[-1][-1]['gamma_1'], 
                                        x0_2_guess           = self.fits[-1][-1]['x0_2'], 
                                        a_2_guess           = self.fits[-1][-1]['a_2'], 
                                        sigma_2_guess           = self.fits[-1][-1]['sigma_2'], 
                                        gamma_2_guess           = self.fits[-1][-1]['gamma_2'], 
                                        b_guess                       = self.fits[-1][-1]['b'], 
                                        bounds          = bounds,
                                        plot                    = plot,
                                        lambda_min_plot         = lambda_min_plot,
                                        lambda_max_plot         = lambda_max_plot
                            )

                        if fit_result==None:
                            print('(i, j) = ('+str(i)+', '+str(j)+')')
                        else: 
                            new_fit_result = {'index_series':i, 'index_spectra':j}
                            new_fit_result.update(fit_result)
                            self.fits[-1].append(new_fit_result)
                            first_fit = False


        if self.fits[-1]!=[]:
            p0 = [self.fits[-1][-1]['x0_1'], self.fits[-1][-1]['a_1'], self.fits[-1][-1]['sigma_1'], self.fits[-1][-1]['gamma_1'], self.fits[-1][-1]['x0_2'], self.fits[-1][-1]['a_2'], self.fits[-1][-1]['sigma_2'], self.fits[-1][-1]['gamma_2'], self.fits[-1][-1]['b']]
        else:
            p0 = None
        return p0


    def fit_gauss(self, lambda0_guess, x_min = None, x_max = None, delta_lambda=.5, sigma_guess = .01, bounds = None, background_type = 'linear', plot=False):
        """
        Fit all spectra with a Gaussian function 

        :param lambda0_guess: (float) [nm]  guess value of the position of the Voigt function
        :param x_min: x value lower bound of the spectrum to fit
        :param x_max: x value upper bound of the spectrum to fit
        :param delta_lambda: (float, nm) range of wavelengths centered around lambda0_guess
        :param sigma_guess: (float) guess value of the standard deviation of the gaussian function
        :param bounds: (dict) dictionary containing the limit values that the function parameters can reach during the fitting procedure. Available keys are
            - "sigma_min"
            - "sigma_max"
        :param background_type: (string) 
            - "none"    : Voigt profile only
            - "const"   : Voigt profile with constant 
            - "linear"  : Voigt profile with linear function
        :param plot: (bool) : if true, plot the spectra and the fitted function
        """

        if x_min ==  None:
            x_min = - numpy.infty
        if x_max ==  None:
            x_max = numpy.infty

        first_fit = True
        self.fits.append([])
        for i in range(len(self.series)):
            for j in range(len(self.series[i].spectra)): # starting with bigger x-values (high powers)
                if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                    if first_fit:
                        fit_result = self.series[i].spectra[j].fit_gauss(
                                lambda0_guess           = lambda0_guess, 
                                delta_lambda            = delta_lambda, 
                                sigma_guess             = sigma_guess,
                                bounds                  = bounds,
                                background_type       = background_type,
                                plot                    = plot
                        )
                    else: # use results from last fit 
                        fit_result = self.series[i].spectra[j].fit_gauss(
                                lambda0_guess           = self.fits[-1][-1]['lambda0'], 
                                delta_lambda            = delta_lambda, 
                                sigma_guess             = self.fits[-1][-1]['sigma'],
                                bounds                  = bounds,
                                background_type       = background_type,
                                plot                    = plot
                        )

                    if fit_result==None:
                        print('(i, j) = ('+str(i)+', '+str(j)+')')
                    else: 
                        new_fit_result = {'index_series':i, 'index_spectra':j}
                        new_fit_result.update(fit_result)
                        self.fits[-1].append(new_fit_result)
                        first_fit = False


    def fit_2_gauss(self, lambda_min_fit, lambda_max_fit, x0_1_guess, a_1_guess, sigma_1_guess, x0_2_guess, a_2_guess, sigma_2_guess,  b_guess, bounds = None, x_min = None, x_max = None, freeze_guesses = False, plot=False, descending=False, lambda_min_plot = None, lambda_max_plot = None):
        """
        Fit spectra with a double Gaussian function 

        :param lambda_min_fit: (float) [nm] minimal wavelength used to fit the function
        :param lambda_max_fit: (float) [nm] maximal wavelength used to fit the function
        :param x0_i_guess: (float) [nm] guess value of the position of the i-th Gaussian function
        :param a_i_guess: (float) guess value of the amplitude of the i-th Gaussian function
        :param sigma_i_guess: (float) guess value of the sigma parameter of the i-th Gaussian function
        :param  b_guess: (float) constant value
        :param bounds: (dict) dictionary containing the minimum and maximum values of the parameters to be fitted. Available bounds are
            - "x0_1_min" (float)
            - "x0_1_max" (float)
            - "x0_2_min" (float)
            - "x0_2_max" (float)
            - "sigma_1_min" (float) 
            - "sigma_2_min" (float) 
        :param x_min: (float) minimal value of x to be fitted
        :param x_max: (float) maximal value of x to be fitted
        :param freeze_guesses: (bool) if true, the same guess values are used for all the fitting
        :param plot: (bool) if true, plot the data with the fitted function
        :param descending: (bool) if true, fit the spectra by decreasing order of associated x value 
        :param lambda_min_plot: (float) [nm] minimal wavelength to be plotted 
        :param lambda_max_plot: (float) [nm] maximal wavelength to be plotted 

        :return: Dictionary of fit result or 'none' if failed
        """

        if x_min ==  None:
            x_min = - numpy.infty
        if x_max ==  None:
            x_max = numpy.infty

        first_fit = True
        self.fits.append([])
        if descending==False: # iterate with x increasing 
            for i in range(len(self.series)-1, -1, -1): 
                for j in range(len(self.series[i].spectra)-1, -1, -1): 
                    if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                        if first_fit or freeze_guesses:
                            fit_result = self.series[i].spectra[j].fit_2_gauss(
                                lambda_min_fit      = lambda_min_fit, 
                                lambda_max_fit      = lambda_max_fit, 
                                x0_1_guess           = x0_1_guess, 
                                a_1_guess           = a_1_guess, 
                                sigma_1_guess           = sigma_1_guess, 
                                x0_2_guess           = x0_2_guess, 
                                a_2_guess           = a_2_guess, 
                                sigma_2_guess           = sigma_2_guess, 
                                b_guess                       = b_guess,
                                bounds          = bounds,
                                plot                    = plot,
                                lambda_min_plot         = lambda_min_plot,
                                lambda_max_plot         = lambda_max_plot
                            )
                        else: # use results from last fit 
                            fit_result = self.series[i].spectra[j].fit_2_gauss(
                                        lambda_min_fit      = lambda_min_fit, 
                                        lambda_max_fit      = lambda_max_fit, 
                                        x0_1_guess           = self.fits[-1][-1]['x0_1'], 
                                        a_1_guess           = self.fits[-1][-1]['a_1'], 
                                        sigma_1_guess           = self.fits[-1][-1]['sigma_1'], 
                                        x0_2_guess           = self.fits[-1][-1]['x0_2'], 
                                        a_2_guess           = self.fits[-1][-1]['a_2'], 
                                        sigma_2_guess           = self.fits[-1][-1]['sigma_2'], 
                                        b_guess                       = self.fits[-1][-1]['b'], 
                                        bounds          = bounds,
                                        plot                    = plot,
                                        lambda_min_plot         = lambda_min_plot,
                                        lambda_max_plot         = lambda_max_plot
                            )
                        if fit_result==None:
                            print('(i, j) = ('+str(i)+', '+str(j)+')')
                        else: 
                            new_fit_result = {'index_series':i, 'index_spectra':j}
                            new_fit_result.update(fit_result)
                            self.fits[-1].append(new_fit_result)
                            first_fit = False
        else: 
            for i in range(len(self.series)):
                for j in range(len(self.series[i].spectra)): # starting with bigger x-values (high powers)
                    if (self.series[i].spectra[j].x > x_min) and (self.series[i].spectra[j].x < x_max):
                        if first_fit or freeze_guesses:
                            fit_result = self.series[i].spectra[j].fit_2_gauss(
                                lambda_min_fit      = lambda_min_fit, 
                                lambda_max_fit      = lambda_max_fit, 
                                x0_1_guess           = x0_1_guess, 
                                a_1_guess           = a_1_guess, 
                                sigma_1_guess           = sigma_1_guess, 
                                x0_2_guess           = x0_2_guess, 
                                a_2_guess           = a_2_guess, 
                                sigma_2_guess           = sigma_2_guess, 
                                b_guess                       = b_guess,
                                bounds          = bounds,
                                plot                    = plot,
                                lambda_min_plot         = lambda_min_plot,
                                lambda_max_plot         = lambda_max_plot
                            )
                        else: # use results from last fit in the same Series as guess values
                            fit_result = self.series[i].spectra[j].fit_2_gauss(
                                        lambda_min_fit      = lambda_min_fit, 
                                        lambda_max_fit      = lambda_max_fit, 
                                        x0_1_guess           = self.fits[-1][-1]['x0_1'], 
                                        a_1_guess           = self.fits[-1][-1]['a_1'], 
                                        sigma_1_guess           = self.fits[-1][-1]['sigma_1'], 
                                        x0_2_guess           = self.fits[-1][-1]['x0_2'], 
                                        a_2_guess           = self.fits[-1][-1]['a_2'], 
                                        sigma_2_guess           = self.fits[-1][-1]['sigma_2'], 
                                        b_guess                       = self.fits[-1][-1]['b'], 
                                        bounds          = bounds,
                                        plot                    = plot,
                                        lambda_min_plot         = lambda_min_plot,
                                        lambda_max_plot         = lambda_max_plot
                            )

                        if fit_result==None:
                            print('(i, j) = ('+str(i)+', '+str(j)+')')
                        else: 
                            new_fit_result = {'index_series':i, 'index_spectra':j}
                            new_fit_result.update(fit_result)
                            self.fits[-1].append(new_fit_result)
                            first_fit = False


        if self.fits[-1]!=[]:
            p0 = [self.fits[-1][-1]['x0_1'], self.fits[-1][-1]['a_1'], self.fits[-1][-1]['sigma_1'], self.fits[-1][-1]['x0_2'], self.fits[-1][-1]['a_2'], self.fits[-1][-1]['sigma_2'], self.fits[-1][-1]['b']]
        else:
            p0 = None
        return p0



    def get_fit_results(self):
        """ 
        Return a pandas DataFrame containing all fit results.
        """
        df = pandas.DataFrame({})
        for fit in self.fits:
            df = df._append(fit, ignore_index=True)
        return df

