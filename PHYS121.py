# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:19:19 2022

@author: jbobowsk
"""

# Import some required modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import hashlib
from matplotlib.pyplot import cm # used to generate a sequence of colours for plotting
from scipy.optimize import curve_fit
from IPython.display import HTML as html_print
from IPython.display import display, Markdown, Latex, clear_output
from datetime import datetime, timedelta
import sympy as sym
import inspect

###############################################################################
# Import a set of modules                                                     #
# - modified 20220817                                                         #
############################################################################### 
# Start the 'packages' function.
def packages():
    global np
    import numpy as np
    global math
    import math
    global plt
    import matplotlib.pyplot as plt
    install_and_import('httpimport')
    with httpimport.remote_repo(['PHYS121'], 'https://cmps-people.ok.ubc.ca/jbobowsk/PHYS_121_Lab/modules'):
        global PHYS121
        import PHYS121
    with httpimport.remote_repo(['uncertainties'], 'https://cmps-people.ok.ubc.ca/jbobowsk/PHYS_121_Lab/modules/uncertainties-3.1.7'):
        global uncertainties
        import uncertainties
    with httpimport.remote_repo(['ipysheet'], 'https://cmps-people.ok.ubc.ca/jbobowsk/PHYS_121_Lab/modules/ipysheet-0.5.0'):
        global ipysheet
        import ipysheet
    with httpimport.remote_repo(['data_entry'], 'https://cmps-people.ok.ubc.ca/jbobowsk/PHYS_121_Lab/modules'):
        global data_entry
        import data_entry
    return
    

###############################################################################
# For printing outputs in colour (copied from Stack Overflow)                 #
# - modified 20220607                                                         #
############################################################################### 
# Start the 'cstr' function.
def cstr(s, color = 'black'):
    return "<text style=color:{}>{}</text>".format(color, s)


###############################################################################
# Force a Float to be Printed w/o using Scientific Notation                   #
# - modified 20220527                                                         #
############################################################################### 
# Start the 'printStr' function.
def printStr(FloatNumber, Precision):
    # Print a float as a string with the number of digits after the decimal
    # determined by 'Precision'.
    return "%0.*f" % (Precision, FloatNumber)


###############################################################################
# Add a Package if it's not Already Installed                                 #
# - modified 20220527                                                         #
############################################################################### 
# Start the 'install_and_import' function.
# Check to see if 'package' is already installed.  If not, then attempt
# to do the install.
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


###############################################################################
# Check to see if required packages are already installed.                    #
# If not, then install them.                                                  #
# - modified 20240624                                                         #
############################################################################### 
# Start the 'Check' function.
def Installer():
    import importlib.util
    import sys
    import subprocess
    cnt = 0
    package_names = ['uncertainties', 'pdfkit', 'PyPDF2']
    for name in package_names:
        spec = importlib.util.find_spec(name)
        if spec is None:
            display(html_print(cstr('Installing some packages ...\n', color = 'red')))
            display(html_print(cstr("After the installation completes, please restart the kernel and then run the 'PHYS231.Installer()' function again before proceeding.\n", color = 'red')))
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', name])
            cnt += 1

    import importlib
    try:
        importlib.import_module('otter')
    except ImportError:
        display(html_print(cstr('Installing some packages ...\n', color = 'red')))
        display(html_print(cstr("After the installation completes, please restart the kernel and then run the 'PHYS231.Installer()' function again before proceeding.\n", color = 'red')))
        import pip
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'otter-grader'])
        #pip.main(['install', 'otter-grader'])
        cnt += 1
    finally:
        globals()['otter'] = importlib.import_module('otter')
    
    if cnt == 0:
        display(html_print(cstr('All packages already installed. Please proceed.', color = 'black')))
    else:
        display(html_print(cstr("\n Some packages were installed.  Please restart the kernel and then run the 'PHYS231.Installer()' function again before proceeding.", color = 'red')))
        

###############################################################################
# Parsing Formatted Outputs from the Uncertainties Package                    #
# - modified 20220527                                                         #
############################################################################### 
# Start the 'Parse' function.
# The uncertainties package generates strings of the form '(x+/-y)eN' where 
# x and y are floats and N is an integer.  This functions separates out 
# x and y and counts the number of places after the decimal in x.
def Parse(uncertain):
    if uncertain[0] != '(':
        z = uncertain.split('+/-')
        num = float(z[0])
        err = float(z[1])
    else:
        uncertain = uncertain[1:]
        z = uncertain.split('+/-')
        num = float(z[0] + uncertain.split(')')[1])
        err = float(z[1].split(')')[0] + z[1].split(')')[1])
    places = len(z[0].split('.')[1])
    return num, err, places


###############################################################################
# Parsing Numbers in Scientific Notation for LaTeX Formatting                 #
# - modified 20220527                                                         #
############################################################################### 
# Start the 'eParse' function.
# This fuction determines the coefficient and power of a number of the form
# xeN, where x is the coefficient and N is the power.  If the number is not
# already expressed in scientific notation, this function will sill determine
# the value of the coefficient and power as if the number was expressed in 
# scientific notation.
def eParse(num, places):
    if isinstance(num, (float, int)) == False:
        print("'num' must be an integer or float.")
    else:
        z = str(num).split('e')
        coeff = float(z[0])
        if len(z) == 1:
            power = np.log10(abs(num)) - np.log10(abs(num)) % 1
            coeff = round(num/10**power, places) # Used to eliminate results such as 1.234000000000000001
        else:
            power = z[1]
    return coeff, int(power)


###############################################################################
# Producing Scatter Plots                                                     #
# - modified 20220530                                                         #
############################################################################### 
# Start the 'Scatter' function.
def Scatter(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = '', fill = False, show = True):
    fig = ''
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData) and xData != []:
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    elif fill != True and fill != False:
        display(html_print(cstr("The 'fill' parameter must be set to either True or False.", color = 'magenta')))
    elif show != True and show != False:
        display(html_print(cstr("The 'show' parameter must be set to either True or False.", color = 'magenta')))
    else:
        fig = plt.figure(figsize=(5, 5), dpi=100) # create a square figure.
        ax = fig.add_subplot(111)
        if len(xData) == 0:
            xData = np.arange(1, len(yData) + 1, 1)
            xlabel = 'trial number'
            xUnits = ''
        if len(yErrors) == 0:
            # plot without error bars
            plt.plot(xData, yData, 'ko', markersize = 6,\
                        markeredgecolor = 'b',\
                        markerfacecolor = 'r')
        else:
            # plot with error bars
            plt.errorbar(xData, yData, yErrors, fmt = 'ko', markersize = 6,\
                        markeredgecolor = 'b',\
                        markerfacecolor = 'r',\
                        capsize = 6)
        
        # Used to add shading around a best-fit line.  The fill is determined
        # by the uncertainties in the parameters.
        if fill == True:
            # If a fill is requested and there are no error bars, generate a list of errors
            # that will equally weight all of the points.
            if len(yErrors) == 0:
                for i in range(len(yData)):
                    yErrors.append(1)
            install_and_import('lmfit') # install 'lmfit' if its not already installed
            # The following lines are neede to determine the shading (copied from Firas Moosvi) 
            from lmfit.models import LinearModel  # import the LinearModel from `lmfit` package
            model = LinearModel()
            params = model.guess(yData, xData)
            result = model.fit(yData, params, x = np.array(xData), weights = 1/np.array(yErrors))
            # Calculate parameter uncertainty
            delmodel = result.eval_uncertainty(x = np.array(xData))
                
        if xUnits != '':
            xlabel = xlabel + ' (' + xUnits + ')' # Add units if provided.
        if yUnits != '':
            ylabel = ylabel + ' (' + yUnits + ')' # Add units if provided.
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # Used to make the plot square
        if fill == True:
            # Used to calculate the fill shading
            ax.fill_between(xData, result.best_fit - delmodel, result.best_fit + delmodel, alpha=0.2)
        if show == True:
            plt.show()
    return fig
    

###############################################################################
# Produce Multiple Scatter Plots                                              #
# - modified 20220530                                                         #
############################################################################### 
# Start the 'MultiScatter' function.
def MultiScatter(DataArray, xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    fig = ''
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(DataArray) == 0:
        display(html_print(cstr("The 'DataArray' list must not be empty.", color = 'magenta')))
    elif all(isinstance(x, list) or type(x).__module__ == np.__name__ for x in DataArray) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The 'DataArray' must be a list of lists.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        for i in range(len(DataArray)):
            if len(DataArray[i]) != 2 and len(DataArray[i]) != 3:
                display(html_print(cstr("The elements of 'DataArray' must be lists of length 2 or 3.  Element " + str(i + 1) + ' does not satisfy this requirement.', color = 'magenta')))
                return fig
            elif all(isinstance(x, (list, pd.core.series.Series)) or type(x).__module__ == np.__name__ for x in DataArray[i]) != True: # Is dataArray a list of lists or arrays
                display(html_print(cstr("The elements of 'DataArray' must be a list of lists.  Element " +  str(i + 1) + ' does not satisfy this requirement.', color = 'magenta')))
                return fig
            elif len(DataArray[i]) == 2:
                if len(DataArray[i][0]) != len(DataArray[i][1]):
                    display(html_print(cstr("In element " + str(i + 1) + " of 'DataArray', the x- and y-datasets are different lengths.", color = 'magenta')))
                    return fig
                elif len(DataArray[i]) == 3:
                    if len(DataArray[i][0]) != len(DataArray[i][1]) or len(DataArray[i][0]) != len(DataArray[i][2]):
                        display(html_print(cstr("In element " + str(i + 1) + " of 'DataArray', the x- y-, and y-error datasets are different lengths.", color = 'magenta')))
                        return fig
        if  type(DataArray).__module__ == np.__name__: # Check to see if DataArray is an array.  If it is, convert to a list.
            DataArray = DataArray.tolist()
        for i in range(len(DataArray)):
            if type(DataArray[i]).__module__ == np.__name__:
                DataArray[i] = DataArray[i].tolist()
            for j in range(len(DataArray[i])):
                if type(DataArray[i][j]).__module__ == np.__name__ or isinstance(DataArray[i][j], pd.core.series.Series):
                    DataArray[i][j] = DataArray[i][j].tolist()
        # Convert DataArray to a single master list
        masterList = sum(sum(DataArray,[]),[])
        if all(isinstance(x, (int, float)) for x in masterList) != True:
            display(html_print(cstr('All elements of x- and y-data and y-errors must be integers or floats.', color = 'magenta')))
            return fig
        
        fig = plt.figure(figsize=(5, 5), dpi=100) # create a square figure.       
        ax = fig.add_subplot(111)
        colour = iter(cm.rainbow(np.linspace(0, 1, len(DataArray))))
        
        for i in range(len(DataArray)):
            c = next(colour)
            if len(DataArray[i]) == 2:
                # plot without error bars
                plt.plot(DataArray[i][0], DataArray[i][1], 'o', color = 'k', markersize = 6,\
                        markeredgecolor = 'b',\
                        markerfacecolor = c)
            elif len(DataArray[i]) == 3:
                # plot with error bars
                plt.errorbar(DataArray[i][0], DataArray[i][1], DataArray[i][2], fmt = 'o', color = 'k', markersize = 6,\
                        markeredgecolor = 'b',\
                        markerfacecolor = c,\
                        capsize = 6)
            
        if xUnits != '':
            xlabel = xlabel + ' (' + xUnits + ')' # Add units if provided.
        if yUnits != '':
            ylabel = ylabel + ' (' + yUnits + ')' # Add units if provided.
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # Used to make the plot square
        plt.show()
    return fig


###############################################################################
# Weighted & Unweighted Linear Fits                                           #
# - modified 20220621                                                         #
############################################################################### 
# Start the 'LinearFit' function.
def LinearFit(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = '', fill = False):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    Slope = ''
    Yintercept = ''
    errSlope = ''
    errYintercept = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    elif fill != True and fill != False:
        display(html_print(cstr("The 'fill' parameter must be set to either True or False.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def linearFunc(x, intercept, slope):
            y = slope*x + intercept
            return y
        
        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        display(Markdown('$y = m\,x + b$'))
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(linearFunc, xData, yData)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(linearFunc, xData, yData, sigma = yErrors)
            display(Markdown('This is a **WEIGHTED** fit.'))

        Slope = a_fit[1]
        errSlope = np.sqrt(np.diag(cov))[1]
        Yintercept = a_fit[0]
        errYintercept = np.sqrt(np.diag(cov))[0]

        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        m = uncertainties.ufloat(Slope, errSlope)
        b = uncertainties.ufloat(Yintercept, errYintercept)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
            my_dict = {'slope' :{'':'$m =$', 'Value': '{:0.2ug}'.format(m), 'Units': yUnits + '/' + xUnits},
                       '$y$-intercept':{'':'$b =$', 'Value': '{:0.2ug}'.format(b), 'Units': yUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = {'slope' :{'':'$m =$', 'Value': '{:0.2ug}'.format(m), 'Units': '1/' + xUnits},
              '$y$-intercept':{'':'$b =$', 'Value': '{:0.2ug}'.format(b), 'Units': yUnits}}
        elif xUnits == '' and yUnits != '':
            my_dict = {'slope' :{'':'$m =$', 'Value': '{:0.2ug}'.format(m), 'Units': yUnits},
              '$y$-intercept':{'':'$b =$', 'Value': '{:0.2ug}'.format(b), 'Units': yUnits}}
        else:
            my_dict = {'slope' :{'':'$m =$', 'Value': '{:0.2ug}'.format(m)},
              '$y$-intercept':{'':'$b =$', 'Value': '{:0.2ug}'.format(b)}}

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, fill, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)
        plt.plot(xx, fitFcn(xx), 'k-')

        # Show the final plot.
        plt.show()
    return Slope, Yintercept, errSlope, errYintercept, fig
        
        
        
###############################################################################
# Weighted & Unweighted Power Law Fits                                        #
# - modified 20230221                                                         #
############################################################################### 
# Start the 'PowerLaw' function.
def PowerLaw(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    Coeff = ''
    Power = ''
    Offset = ''
    errCoeff = ''
    errPower = ''
    errOffset = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def PowerFunc(x, coeff, power, offset):
            y = coeff*x**power + offset
            return y
        
        # Find the minimum y value.  Will scale yData by the minimum value of y so that the nonlinear fit doesn't fail or require good initial guesses at the parameter values.
        ymax = np.max(yData)

        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        if xUnits == '':
            display(Markdown('$y = A\,x^N + C$'))
        else:
            display(Markdown('$y = A\,(x/1$ ' + xUnits + '$)^N\,+ \,C$'))
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(PowerFunc, xData, yData/ymax)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(PowerFunc, xData, yData/ymax, sigma = yErrors/ymax)
            display(Markdown('This is a **WEIGHTED** fit.'))

        Coeff = a_fit[0]*ymax
        errCoeff = np.sqrt(np.diag(cov))[0]*ymax
        Power = a_fit[1]
        errPower = np.sqrt(np.diag(cov))[1]
        Offset = a_fit[2]*ymax
        errOffset = np.sqrt(np.diag(cov))[2]*ymax

        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        A = uncertainties.ufloat(Coeff, errCoeff)
        N = uncertainties.ufloat(Power, errPower)
        C = uncertainties.ufloat(Offset, errOffset)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
#            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits + '/' + xUnits + eval(r'"\u00b' + str(Power) + '"')},
#                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
#                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}           
        elif xUnits == '' and yUnits != '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}} 
        else:
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A)},
                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N)},
                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C)}} 

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        #fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)
        plt.plot(xx, Coeff*xx**Power + Offset, 'k-')

        # Show the final plot.
        plt.show()
    return Coeff, Power, Offset, errCoeff, errPower, errOffset, fig



###############################################################################
# Weighted & Unweighted RC Charing Fits                                       #
# - modified 20220711                                                         #
############################################################################### 
# Start the 'Charging' function.
def Charging(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    A_fit = ''
    tau_fit = ''
    errA = ''
    errTau = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def ChargingFcn(x, A, tau):
            y = A*(1 - np.exp(-x/tau))
            return y
        
        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        display(Markdown('$y = A(1 - e^{-x/ \\tau})$'))
    
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(ChargingFcn, np.array(xData)/max(xData), np.array(yData)/max(yData))
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(ChargingFcn, np.array(xData)/max(xData), np.array(yData)/max(yData), sigma = yErrors)
            display(Markdown('This is a **WEIGHTED** fit.'))

        A_fit = a_fit[0]*max(yData)
        errA = np.sqrt(np.diag(cov))[0]*max(yData)
        tau_fit = a_fit[1]*max(xData)
        errTau = np.sqrt(np.diag(cov))[1]*max(xData)
        
        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        A = uncertainties.ufloat(A_fit, errA)
        tau = uncertainties.ufloat(tau_fit, errTau)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
#            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits + '/' + xUnits + eval(r'"\u00b' + str(Power) + '"')},
#                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
#                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'time constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(tau), 'Units': xUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'time constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(tau), 'Units': xUnits}}       
        elif xUnits == '' and yUnits != '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'time constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(tau), 'Units': xUnits}}
        else:
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A)},
                       'time constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(tau)}}

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        #fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)
        plt.plot(xx, A_fit*(1 - np.exp(-xx/tau_fit)), 'k-')

        # Show the final plot.
        plt.show()
    return A_fit, tau_fit, errA, errTau, fig

###############################################################################
# Weighted & Unweighted Sine                                                  #
# - modified 20230320                                                         #
############################################################################### 
# Start the 'Sine' function.
def Sine(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    Coeff = ''
    Power = ''
    Offset = ''
    errCoeff = ''
    errPower = ''
    errOffset = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def SineFcn(x, coeff, period, phase):
            y = coeff*np.sin(2*np.pi/period*x + phase)
            return y
        
        # Find the minimum y value.  Will scale yData by the minimum value of y so that the nonlinear fit doesn't fail or require good initial guesses at the parameter values.
        ymax = np.max(yData)
        T_Est = 1.41 #  period in s.

        start = (ymax, T_Est, 0)


        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        if xUnits == '':
            display(Markdown('$y = A\,\sin(2\pi/T x + \phi)$'))
        else:
            display(Markdown('$y = A\,\sin(2\pi/T x + \phi)$'))
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(SineFcn, xData, yData, p0 = start)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(SineFcn, xData, yData, sigma = yErrors, p0 = start)
            display(Markdown('This is a **WEIGHTED** fit.'))

        coeff = a_fit[0]
        errCoeff = np.sqrt(np.diag(cov))[0]
        period = a_fit[1]
        errPeriod = np.sqrt(np.diag(cov))[1]
        phase = a_fit[2]
        errPhase = np.sqrt(np.diag(cov))[2]

        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        A = uncertainties.ufloat(coeff, errCoeff)
        T = uncertainties.ufloat(period, errPeriod)
        phi = uncertainties.ufloat(phase, errPhase)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
#            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits + '/' + xUnits + eval(r'"\u00b' + str(Power) + '"')},
#                       'period':{'':'$T =$', 'Value': '{:0.2ug}'.format(T), 'Units': xUnits},
#                       'phase':{'':'$\phi =$', 'Value': '{:0.2ug}'.format(phi), 'Units': 'rad'}}
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'period':{'':'$T =$', 'Value': '{:0.2ug}'.format(T), 'Units': xUnits},
                       'phase':{'':'$\phi =$', 'Value': '{:0.2ug}'.format(phi), 'Units': 'rad'}}
        elif xUnits != '' and yUnits == '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'period':{'':'$T =$', 'Value': '{:0.2ug}'.format(T), 'Units': xUnits},
                       'phase':{'':'$\phi =$', 'Value': '{:0.2ug}'.format(phi), 'Units': 'rad'}}           
        elif xUnits == '' and yUnits != '':
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'period':{'':'$T =$', 'Value': '{:0.2ug}'.format(T), 'Units': xUnits},
                       'phase':{'':'$\phi =$', 'Value': '{:0.2ug}'.format(phi), 'Units': 'rad'}} 
        else:
            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A)},
                       'period':{'':'$T =$', 'Value': '{:0.2ug}'.format(T)},
                       'phase':{'':'$\phi =$', 'Value': '{:0.2ug}'.format(phi)}} 

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        #fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)
        plt.plot(xx, coeff*np.sin(2*np.pi/period*xx + phase), 'k-')

        # Show the final plot.
        plt.show()
    return coeff, period, phase, errCoeff, errPeriod, errPhase, fig


###############################################################################
# Magnetic Braking Nonlinear Fits                                             #
# - modified 20220710                                                         #
############################################################################### 
# Start the 'Braking' function.
def Braking(xData, yData, yErrors = [], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    vterm = ''
    tau = ''
    errvterm = ''
    errtau = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def BrakingFcn(x, tau, vterm):
            y = vterm*tau*((x/tau) - 1 + np.exp(-x/tau))
            return y
        
        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        display(Markdown('$y = v_\mathrm{T}\\tau\left[\dfrac{x}{\\tau} - 1 + e^{-x/ \\tau}\\right]$'))

        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(BrakingFcn, xData, yData)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(BrakingFcn, xData, yData, sigma = yErrors)
            display(Markdown('This is a **WEIGHTED** fit.'))

        Vterm = a_fit[1]
        errVterm = np.sqrt(np.diag(cov))[1]
        Tau = a_fit[0]
        errTau = np.sqrt(np.diag(cov))[0]

        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        v = uncertainties.ufloat(Vterm, errVterm)
        T = uncertainties.ufloat(Tau, errTau)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
            my_dict = {'Terminal Velocity' :{'':'$v_\mathrm{t} =$', 'Value': '{:0.2ug}'.format(v), 'Units': yUnits + '/' + xUnits},
                       'Time Constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(T), 'Units': yUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = {'Terminal Velocity' :{'':'$v_\mathrm{t} =$', 'Value': '{:0.2ug}'.format(v), 'Units': '1/' + xUnits},
              'Time Constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(T), 'Units': yUnits}}
        elif xUnits == '' and yUnits != '':
            my_dict = {'Terminal Velocity' :{'':'$v_\mathrm{t} =$', 'Value': '{:0.2ug}'.format(v), 'Units': yUnits},
              'Time Constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(T), 'Units': yUnits}}
        else:
            my_dict = {'Terminal Velocity' :{'':'$v_\mathrm{t} =$', 'Value': '{:0.2ug}'.format(v)},
              'Time Constant':{'':'$\tau =$', 'Value': '{:0.2ug}'.format(T)}}

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
                
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)
        # Generate the best-fit line. 
        fitFcn = Vterm*Tau*((xx/Tau) - 1 + np.exp(-xx/Tau))

        plt.plot(xx, fitFcn, 'k-')

        # Show the final plot.
        plt.show()
    return Vterm, Tau, errVterm, errTau, fig
    
        
###############################################################################
# Histograms & Statistics                                                     #
# - modified 20220529                                                         #
###############################################################################        
# Start the 'Statisitics' function.
def Statistics(data, nbins = 10, xlabel = 'x-axis', xUnits = '', normalized = False):
    counts = ''
    centres = ''
    average = ''
    stdDev = ''
    stdError = ''
    fig = ''
    if len(data)==0:
        display(html_print(cstr("The 'data' list must not be empty.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in data) != True:
        display(html_print(cstr("All elements of the 'data' list must be floats or integers.", color = 'magenta')))
    elif isinstance(nbins, int) == False or nbins < 0:
        display(html_print(cstr("'nbins' must be a positive integer.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif normalized != True and normalized != False:
        display(html_print(cstr("The 'normalized' parameter must be set to either True or False.", color = 'magenta')))
    else:
        # Determine the boundaries of the various histogram bins.
        binwidth = (np.max(data) - np.min(data))/nbins
        boundaries = np.arange(np.min(data), np.max(data) + binwidth, binwidth)
        
        # Prepare a square figure.
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot to histogram
        counts, edges, patches = plt.hist(data, bins = boundaries, color='lightskyblue', edgecolor='k', density = normalized)
        
        # Use the output from the histogram plot to determine the positions of the bin centres.
        centres = edges[0:len(counts)] + binwidth/2
        
        # Calculate some basic statistics
        average = np.mean(data)
        stdDev = np.std(data,  ddof=1)
        stdError = stdDev/np.sqrt(len(data))
        
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties
        
        # Use the uncertainties package and Parsing functions to help format the statistics 
        x = uncertainties.ufloat(average, stdError)
        y = '{:0.2ue}'.format(x)
        num, stdError, places = Parse(y)
        stdDev = float('{:0.3g}'.format(stdDev))
        coeff, power = eParse(num, places)
        print('')
        
        # Display some nicely-formatted results
        if abs(power) < 3:        
            stdError = printStr(stdError, places - power)
            display(Latex(f'$N = {len(data)}$.'))
            display(Latex(f'The average of the data is $\mu = {num}\, \mathrm{{ {xUnits} }}.$'))
            display(Latex(f'The standard deviation of the data is $\sigma = {stdDev}\, \mathrm{{ {xUnits} }}.$'))
            display(Latex(f'The standard error of the data is $\sigma_\mu = \sigma/\sqrt{{N}} = \mathrm{stdError}\, \mathrm{{ {xUnits} }}.$'))
        else:
            stdDevPrint = round(stdDev/10**power, places)
            stdErrorPrint = round(stdError/10**power, places)
            display(Latex(f'$N = {len(data)}$.'))
            display(Latex(f'The average of the data is $\mu = {coeff}' + r'\times' +  f'10^{{ {power} }}\, \mathrm{{ {xUnits} }}.$'))
            display(Latex(f'The standard deviation of the data is $\sigma = {stdDevPrint}' + r'\times' + f'10^{{ {power} }}\, \mathrm{{ {xUnits} }}.$'))
            display(Latex(f'The standard error of the data is $\sigma_\mu = \sigma/\sqrt{{N}} = {stdErrorPrint}' + r'\times' + f'10^{{ {power} }}\, \mathrm{{ {xUnits} }}.$'))
        
        # Add units if they were provided.
        if xUnits != '':
            xlabel = xlabel + ' (' + xUnits + ')'
        
        # Format and show the plot.
        plt.xlabel(xlabel)
        if normalized == True:
            plt.ylabel('Normalized Counts')
        else:
            plt.ylabel('Counts')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plt.show()
        stdError = float(stdError)
    return counts, centres, average, stdDev, stdError, fig


###############################################################################
# Overlay Multiple Histograms                                                 #
# - modified 20220801                                                         #
###############################################################################        
# Start the 'HistOverlay' function.
def HistOverlay(dataArray, nbins = 10, xlabel = 'x-axis', xUnits = '',  normalized = True, transparency = 0.75):
    countsArray = ''
    centresArray = ''
    fig = ''
    if len(dataArray)==0:
        display(html_print(cstr("The 'dataArray' list must not be empty.", color = 'magenta')))
    elif all(isinstance(x, list) or type(x).__module__ == np.__name__ for x in dataArray) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The 'dataArray' must be a list of lists.", color = 'magenta')))
    elif isinstance(nbins, int) == False or nbins < 0:
        display(html_print(cstr("'nbins' must be a positive integer.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif normalized != True and normalized != False:
        display(html_print(cstr("The 'normalized' parameter must be set to either True or False.", color = 'magenta')))
    elif isinstance(transparency, int) == False and isinstance(transparency, float) == False:
        display(html_print(cstr("The 'transparency' parameter must be a number between 0 (completely transparent) and 1 (opaque).", color = 'magenta')))
    elif transparency < 0 or transparency > 1:
        display(html_print(cstr("The 'transparency' parameter must be a number between 0 (completely transparent) and 1 (opaque).", color = 'magenta')))
    else:
        # Check to see if dataArray is a numpy array.  It it is, convert to a list.
        if  type(dataArray).__module__ == np.__name__:
            dataArray = dataArray.tolist()
        # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
        for i in range(len(dataArray)):
            if  type(dataArray[i]).__module__ == np.__name__:
                    dataArray[i] = dataArray[i].tolist()
        # Generate a sequence of colours used to plot the the multiple histograms.
        colour = iter(cm.rainbow(np.linspace(0, 1, len(dataArray))))
        tot = sum(dataArray, []) # Combine the list of lists into a single large list
        if all(isinstance(x, (int, float)) for x in tot) != True:
            display(html_print(cstr('All elements of the provided data must be integers or floats.', color = 'magenta')))
            return countsArray, centresArray, fig
        # Calculate the boundaries of the histogram bins
        binwidth = (np.max(tot) - np.min(tot))/nbins
        boundaries = np.arange(np.min(tot), np.max(tot) + binwidth, binwidth)
        
        # Plot the histograms and store the outputs in lists.
        countsArray = []
        centresArray = []
        fig = plt.figure()
        for i in range(len(dataArray)): 
            c = next(colour)
            c[3] = transparency
            counts, edges, patches = plt.hist(dataArray[i], bins = boundaries, fc = c, edgecolor='k', density = normalized);
            centres = edges[0:len(counts)] + binwidth/2
            countsArray.append(counts)
            centresArray.append(centres)
        plt.close(fig)

        # Perpare a squre figure
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)

        plt.hist(dataArray, bins = boundaries, edgecolor='k', density = normalized)

        # Add units if they were provided.
        if xUnits != '':
            xlabel = xlabel + ' (' + xUnits + ')'
            
        # Format and show the plot.
        plt.xlabel(xlabel)
        if normalized == True:
            plt.ylabel('Normalized Counts')
        else:
            plt.ylabel('Counts')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # Make the plot square
        plt.show()
    return countsArray, centresArray, fig


###############################################################################
# Overlap Multiple Histograms                                                 #
# - modified 20220801                                                         #
###############################################################################        
# Start the 'HistOverlap' function.
def HistOverlap(dataArray, nbins = 10, xlabel = 'x-axis', xUnits = '',  normalized = True, transparency = 0.75):
    countsArray = ''
    centresArray = ''
    fig = ''
    if len(dataArray)==0:
        display(html_print(cstr("The 'dataArray' list must not be empty.", color = 'magenta')))
    elif all(isinstance(x, list) or type(x).__module__ == np.__name__ for x in dataArray) != True: # Is dataArray a list of lists or arrays?
        display(html_print(cstr("The 'dataArray' must be a list of lists.", color = 'magenta')))
    elif isinstance(nbins, int) == False or nbins < 0:
        display(html_print(cstr("'nbins' must be a positive integer.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif normalized != True and normalized != False:
        display(html_print(cstr("The 'normalized' parameter must be set to either True or False.", color = 'magenta')))
    elif isinstance(transparency, int) == False and isinstance(transparency, float) == False:
        display(html_print(cstr("The 'transparency' parameter must be a number between 0 (completely transparent) and 1 (opaque).", color = 'magenta')))
    elif transparency < 0 or transparency > 1:
        display(html_print(cstr("The 'transparency' parameter must be a number between 0 (completely transparent) and 1 (opaque).", color = 'magenta')))
    else:
        # Check to see if dataArray is a numpy array.  It it is, convert to a list.
        if  type(dataArray).__module__ == np.__name__:
            dataArray = dataArray.tolist()
        # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
        for i in range(len(dataArray)):
            if  type(dataArray[i]).__module__ == np.__name__:
                    dataArray[i] = dataArray[i].tolist()
        # Generate a sequence of colours used to plot the the multiple histograms.
        colour = iter(cm.rainbow(np.linspace(0, 1, len(dataArray))))
        tot = sum(dataArray, []) # Combine the list of lists into a single large list
        if all(isinstance(x, (int, float)) for x in tot) != True:
            display(html_print(cstr('All elements of the provided data must be integers or floats.', color = 'magenta')))
            return countsArray, centresArray, fig
        # Calculate the boundaries of the histogram bins
        binwidth = (np.max(tot) - np.min(tot))/nbins
        boundaries = np.arange(np.min(tot), np.max(tot) + binwidth, binwidth)
        
        # Perpare a squre figure
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the histograms and store the outputs in lists.
        countsArray = []
        centresArray = []
        for i in range(len(dataArray)): 
            c = next(colour)
            c[3] = transparency
            counts, edges, patches = plt.hist(dataArray[i], bins = boundaries, fc = c, edgecolor='k', density = normalized);
            centres = edges[0:len(counts)] + binwidth/2
            countsArray.append(counts)
            centresArray.append(centres)

        # Add units if they were provided.
        if xUnits != '':
            xlabel = xlabel + ' (' + xUnits + ')'
            
        # Format and show the plot.
        plt.xlabel(xlabel)
        if normalized == True:
            plt.ylabel('Normalized Counts')
        else:
            plt.ylabel('Counts')
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box') # Make the plot square
        plt.show()
    return countsArray, centresArray, fig


###############################################################################
# Import Image & Add a Caption                                                #
# - modified 20230205                                                         #
###############################################################################        
# Start the 'Import Image' function.
def ImportImage(filename, caption = '', rotation = 0):
    from os.path import exists as file_exists
    fig = ''
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.eps')) == False:
        display(html_print(cstr('The file type must be one of the following: png, jpg, jpeg, gif, eps.', color = 'magenta')))
    elif file_exists(filename) == False:
        display(html_print(cstr("The file '" + filename + "'does not exist.  Check the file name and ensure that it is in the same directory as your current Jupyter Notebook.", color = 'magenta')))
    elif isinstance(caption, str) == False:
        display(html_print(cstr('The caption must be a string.', color = 'magenta')))
    elif "\\" in r"%r" % caption:
        display(html_print(cstr(r"The caption cannot contain backslashes '\'.", color = 'magenta')))
    elif isinstance(rotation, (float, int)) == False:
        display(html_print(cstr('The rotational angle must be a float or integer.  It represents the rotation angle in degrees.', color = 'magenta')))
    else:
        from PIL import Image
        fig = plt.figure(figsize=(12, 8), dpi=100) # create a square figure.
        img = Image.open(filename) # Open the file
        img = img.rotate(rotation, expand = 1) # Rotate the file.
        plt.imshow(img)
        plt.axis('off') # Remove the axes from the 'plot'.
        plt.text(0, 0,'%s' %caption, size = 14, color = "k") # Add the caption below the image.
        plt.show() # Show the image.
    return
    
    
###############################################################################
# Enter Data using a Spreadsheet-like Environment                             #
# Makes use of data_entry which was written by Carl Michal (UBCV)             #
# - modified 20220621                                                         #
###############################################################################       
# Check to see if ipysheet is installed.
def Spreadsheet(csvName):
    if isinstance(csvName, str) == False:
        display(html_print(cstr("'csvName' must be a string.", color = 'magenta')))
    else:
        install_and_import('ipysheet')
        import data_entry
        data_entry.sheet(csvName)
    return


###############################################################################
# Produce contour and vector field plots                                      #
# - modified 20230205                                                         #
###############################################################################       
# Check to see if ipysheet is installed.
def Mapping(x_coord, y_coord, potential, graphNum = 0, vectorField = True, fig_file_name = 'figure.png'):
    
    import matplotlib.pyplot as plt
    import scipy.interpolate
    from matplotlib.pyplot import figure
    from scipy import interpolate
    from IPython.display import HTML as html_print
    from scipy.interpolate import interp1d

    fig = ''
    
    # Check for errors in the inputs.
    if len(x_coord) != len(y_coord):
        display(html_print(cstr('The length of x_coord (' + str(len(x_coord)) + ') is not equal to the length of y_coord (' + str(len(y_coord)) + ').', color = 'magenta')))
    elif len(x_coord) != len(potential):  
        display(html_print(cstr('The length of x_coord (' + str(len(x_coord)) + ') is not equal to the length of potential (' + str(len(potential)) + ').', color = 'magenta')))
    elif len(y_coord) != len(potential):  
        display(html_print(cstr('The length of y_coord (' + str(len(y_coord)) + ') is not equal to the length of potential (' + str(len(potential)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in x_coord) != True:
        display(html_print(cstr("The elements of 'x_coord' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in y_coord) != True:
        display(html_print(cstr("The elements of 'y_coord' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in potential) != True:
        display(html_print(cstr("The elements of 'potential' must be integers or floats.", color = 'magenta')))
    elif isinstance(graphNum, int) == False:
        display(html_print(cstr("'graphNum' must be an integer from 1 to 7 corresponding to the graph number on your board.", color = 'magenta')))
    elif isinstance(fig_file_name, str) == False:
        display(html_print(cstr("'fig_file_name' must be a string.", color = 'magenta')))
    else:
        
        x = x_coord
        y = y_coord
        V = potential
 
        # First, interpolate the data that was entered.  There's no extrapolation here.
        # We're using this method initially because it does not require a regular grid of input data and the data does not 
        # need to be sorted.
        N = 200
        xi = np.linspace(1, 24, N)
        yi = np.linspace(1, 18, N)
        zi = scipy.interpolate.griddata((x, y), V, (xi[None,:], yi[:,None]), method = 'cubic')
        
        # Check if there are any 'NaN' entries in zi, where 'NaN' means 'not a number'.  
        # If a 'NaN' is found, show the current potential map and go no further.  The 'NaN' will break the next step 
        # of the function.
        if np.isnan(np.min(zi)) == True:
            fig = plt.figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)
            plt.contourf(xi, yi, zi, levels = np.arange(-0.5, 11, 0.1), cmap='RdYlBu_r');
            plt.colorbar(ticks = np.linspace(0, 10, 11))
            plt.contour(xi, yi, zi, levels = np.arange(-0.5, 11, 0.5), colors = 'dimgray', linewidths = 0.5);
            plt.axis((0, 25, 0, 20))
            plt.xlabel('x (cm)')
            plt.ylabel('y (cm)')
            ax.set_xticks([0, 5, 10, 15, 20, 25])
            ax.set_yticks([0, 5, 10, 15, 20])
            plt.gca().set_aspect(1)
            print('There are gaps in the data.  Please collect more data to fill in parts of the plot that are empty.')
        else:
        
            # The interpolated data follows a grid and is sorted.  Now do a second interpolation that does include exprapolation.
            # The extrapolation will fill the plot from zero to 25 in the x direction and from zero to 20 in the y direction.
            #xx, yy = np.meshgrid(xi, yi)
            #Z = np.zeros(np.shape(xx))

            #f = interpolate.interp2d(xi, yi, zi, kind='cubic') -- interp2d is deprecated
            f = scipy.interpolate.RectBivariateSpline(yi, xi, zi) # Replaced 20250130
            
            xnew = np.arange(0, 25.025, 0.025)
            ynew = np.arange(0, 20.02, 0.02)
            
            Xnew, Ynew = np.meshgrid(xnew, ynew)
            Znew = f(ynew, xnew)
            
            if graphNum == 6:
                btm = -1
            else:
                btm = -0.5
            
            # Generate the contour plot.
            fig = plt.figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)
            plt.contourf(Xnew, Ynew, Znew, levels = np.arange(btm, 11, 0.1), cmap='RdYlBu_r');
            plt.colorbar(ticks = np.linspace(0, 10, 11))
            CS = plt.contour(Xnew, Ynew, Znew, levels = np.arange(btm, 11, 0.5), colors = 'dimgray', linewidths = 0.5);
            # ax.clabel(CS, CS.levels, inline=True, fontsize=10)
            plt.axis((0, 25, 0, 20))
            plt.xlabel('x (cm)')
            plt.ylabel('y (cm)')
            ax.set_xticks([0, 5, 10, 15, 20, 25])
            ax.set_yticks([0, 5, 10, 15, 20])
            plt.gca().set_aspect(1)

            Sc = 2
            fact = 1
            # The following if statement is used to mark the positions of the electrodes and/or conductive paint.
            if graphNum == 1:
                Sc = 2
                fact = 1
                plt.plot(6.5,10, marker = 'o', markersize = 7, markerfacecolor = 'r', markeredgecolor = 'white', linewidth = 3)
                plt.plot(18.5,10, marker = 'o', markersize = 7, markerfacecolor = 'k', markeredgecolor = 'white', linewidth = 3)
            elif graphNum == 2:
                Sc = 3
                fact = 1
                plt.plot(3.7,10, marker = 'o', markersize = 7, markerfacecolor = 'r', markeredgecolor = 'white', linewidth = 3)
                plt.plot(21.5,10, marker = 'o', markersize = 7, markerfacecolor = 'k', markeredgecolor = 'white', linewidth = 3)
                plt.gca().add_patch(plt.Circle((12.5, 10), 5.7, edgecolor='silver', facecolor = 'None', linewidth = 3)) 
            elif graphNum == 3:
                Sc = 2
                fact = 1
                point1 = [6.5, 4]
                point2 = [6.5, 16]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)
                point3 = [18.7, 4]
                point4 = [18.7, 16]
                x_values = [point3[0], point4[0]]
                y_values = [point3[1], point4[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)    
            elif graphNum == 4:
                Sc = 2
                fact = 1
                point1 = [5.5, 4]
                point2 = [5.5, 16]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth =3)
                # Enter a parametric equation that defines a 'teardrop' shape.
                tt = np.arange(0, 2*np.pi, 0.01)
                x4fcn = -np.cos(tt)*7.3/2 + 13 + 7.5/2
                y4fcn = np.sin(tt)*np.sin(tt/2)*5/1.5 + 10
                #x4 = [13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 20, 19.6, 19, 18.5, 18, 17.5, 17, 16.5, 16, 15.5, 15, 14.5, 14, 13.5, 13]
                #y4 = [10, 9.6, 9.2, 8.8, 8.5, 8.2, 8, 7.8, 7.7, 7.5, 7.4, 7.6, 7.9, 8.3, 9 , 10, 11, 11.7, 12.1, 12.4, 12.6, 12.5, 12.3, 12.2, 12, 11.8, 11.5, 11.2, 10.8, 10.4, 10]
                plt.plot(x4fcn, y4fcn, 'silver', linewidth = 3)
            elif graphNum == 5:
                Sc = 2
                fact = 1
                point1 = [6.5, 4]
                point2 = [6.5, 16]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)
                point3 = [13.2, 4]
                point4 = [18.5, 16]
                x_values = [point3[0], point4[0]]
                y_values = [point3[1], point4[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)      
            elif graphNum == 6:
                Sc = 4
                fact = 1
                point1 = [6.5, 4]
                point2 = [6.5, 16]
                x_values = [point1[0], point2[0]]
                y_values = [point1[1], point2[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)
                point3 = [18.5, 4]
                point4 = [18.5, 16]
                x_values = [point3[0], point4[0]]
                y_values = [point3[1], point4[1]]
                plt.plot(x_values, y_values, color = 'silver', linestyle='solid', linewidth = 3)  
                plt.gca().add_patch(plt.Rectangle((11.5, 8), 2, 4, edgecolor='slategray', facecolor = 'slategray', linewidth = 3, alpha = 0.75))
                plt.gca().add_patch(plt.Rectangle((11.5, 8), 2, 4, edgecolor='k', facecolor = 'None', linewidth = 3))
            elif graphNum == 7:
                Sc = 10
                fact = 2
                plt.gca().add_patch(plt.Circle((12.5, 10), 5.6, edgecolor='silver', facecolor = 'None', linewidth = 3))
                plt.gca().add_patch(plt.Circle((12.5, 10), 1.6, edgecolor='silver', facecolor = 'None', linewidth = 3))
            else:
                print('Enter an integer from 1 to 7 corresponding to the Graph Number on your board to show the positions of the electrodes.')

            if vectorField == True:
                # Calculate the electric field at all the points in the interpolated/extrapolated potential.
                xEle, yEle = np.mgrid[0:25.025:0.025, 0:20.02:0.02]
                zEle = np.transpose(Znew)
                Ex, Ey = np.gradient(zEle, 0.025, 0.02)
                Ex = -Ex
                Ey = -Ey

                # Sample some of the electric field points for plotting.
                x_E = np.zeros(np.shape(xEle))
                y_E = np.zeros(np.shape(xEle))
                ExSub = np.zeros(np.shape(xEle))
                EySub = np.zeros(np.shape(xEle))
                x_cnt = 0
                for i in range(0, np.shape(Ex)[0], int(100/fact)):
                    y_cnt = 0
                    for j in range(0, np.shape(Ey)[1], int(80/fact)):
                        x_E[x_cnt][y_cnt] = xEle[i][j]
                        y_E[x_cnt][y_cnt] = yEle[i][j]
                        ExSub[x_cnt][y_cnt] = Ex[i][j]
                        EySub[x_cnt][y_cnt] = Ey[i][j]
                        if graphNum == 6 and xEle[i][j] >= 11.5 and xEle[i][j] <= 13.5 and yEle[i][j] >= 8 and yEle[i][j] <= 12:
                            x_E[x_cnt][y_cnt] = -1
                            y_E[x_cnt][y_cnt] = -1
                            ExSub[x_cnt][y_cnt] = 0
                            EySub[x_cnt][y_cnt] = 0
                        y_cnt += 1
                    x_cnt += 1
        
                # Plot the electric field vectors.
                plt.quiver(x_E, y_E, ExSub, EySub, scale = Sc, scale_units = 'inches', width = 0.0035, color = 'k')
                plt.savefig(fig_file_name, format='png')

    return

###############################################################################
# Generate a sequence of random integers and then find their product          #
# - modified 20230109                                                         #
###############################################################################       
def printDigits():
    # This randomly choses how many digits the generated number should be
    numDigits = random.randint(25, 35)
    
    # Now we generate a list of random integers numDigits long
    test = False
    while test == False:
        digits = list(np.random.randint(1, 10, numDigits))
        seen = set()
        duplicates = list(set(x for x in digits if x in seen or seen.add(x)))
        if 9 in digits or (3 in digits and 6 in digits) or 3 in duplicates or 6 in duplicates:
            test = True
    
    # Next, we take their product
    product = 1
    digList = []
    for n in digits:
        product = product * int(n)
        digList += [int(n)]
        
    # Print the results
    print(f"Number of digits: {int(numDigits)}\nList of digits: {digList}\nProduct: {int(product)}")
    return
    
###############################################################################
# Determine which digit generated from a product of integers was set to zero  #
# - modified 20230109                                                         #
###############################################################################       
# Find the digit that was set to zero.
def chase(Number):
    length = len(Number) # Determine the length of Number.  Number is a string.
    if Number[0] == '0':
        Number = Number[1:length] # If Number has a leading zero, remove it
    Number = int(Number) # Convert Number into an integer
    while Number >= 10:
        charList = list(str(Number)) # split numbers into individual string digits
        Number = 0
        for i in charList: # Sum the digits
            Number += int(i)
    if Number != 9: # Find the suppressed digit
        Number = 9 - Number
    return Number


###############################################################################
# Change the extension of a string representing the name of a file            #
# - modified 20230205                                                         #
###############################################################################       
def extension(file_names_in, new_ext):
    file_names_out = []
    for f in file_names_in:
        file_name, file_extension = os.path.splitext(f)
        file_names_out.append(file_name + "." + new_ext)
    return file_names_in + file_names_out


###############################################################################
# Hash student answers so that the Otter-Grader grader.check() output         #
# doesn't reveal the correct answer                                           #
# - modified 20230310                                                         #
###############################################################################       
def get_hash(num):
    """Helper function for assessing correctness"""
    return hashlib.md5(str(num).encode()).hexdigest()

###############################################################################
# Hash student answers so that the Otter-Grader grader.check() output         #
# doesn't reveal the correct answer                                           #
# - modified 20240221                                                         #
###############################################################################       
def hasher(num):
    if num == 'a':
        hashcode = '0cc175b9c0f1b6a831c399e269772661'
    elif num == 'b':
        hashcode = '92eb5ffee6ae2fec3ad71c777531578f'
    elif num == 'c':
        hashcode = '4a8a08f09d37b73795649038408b5f33'
    elif num == 'd':
        hashcode = '8277e0910d750195b448797616e091ad'
    elif num == 'e':
        hashcode = 'e1671797c52e15f763380b45e841ec32'
    elif num == 'f':
        hashcode = '8fa14cdd754f91cc6554c9e71929cce7'
    elif num == 'g':
        hashcode = 'b2f5ff47436671b6e533d8dc3614845d'
    elif num == 'h':
        hashcode = '2510c39011c5be704182423e3a695e91'
    elif num == 'i':
        hashcode = '865c0c0b4ab0e063e5caa3387c1a8741'
    elif num == 'j':
        hashcode = '363b122c528f54df4a0446b6bab05515'
    elif num == 'k':
        hashcode = '8ce4b16b22b58894aa86c421e8759df3'
    elif num == 'l':
        hashcode = '2db95e8e1a9267b7a1188556b2013b33'
    elif num == 'm':
        hashcode = '6f8f57715090da2632453988d9a1501b'
    elif num == 'n':
        hashcode = '7b8b965ad4bca0e41ab51de7b31363a1'
    elif num == 'o':
        hashcode = 'd95679752134a2d9eb61dbd7b91c4bcc'
    elif num == 'p':
        hashcode = '83878c91171338902e0fe0fb97a8c47a'
    elif num == 'q':
        hashcode = '7694f4a66316e53c8cdd9d9954bd611d'
    elif num == 'r':
        hashcode = '4b43b0aee35624cd95b910189b3dc231'
    elif num == 's':
        hashcode = '03c7c0ace395d80182db07ae2c30f034'
    elif num == 't':
        hashcode = 'e358efa489f58062f10dd7316b65649e'
    elif num == 'u':
        hashcode = '7b774effe4a349c6dd82ad4f4f21d34c'
    elif num == 'v':
        hashcode = '9e3669d19b675bd57058fd4664205d2a'
    elif num == 'w':
        hashcode = 'f1290186a5d0b1ceab27f4e77c0c5d68'
    elif num == 'x':
        hashcode = '9dd4e461268c8034f5c8564e155c67a6'
    elif num == 'y':
        hashcode = '415290769594460e2e485922904f345d'
    elif num == 'z':
        hashcode = 'fbade9e36a3f36d3d676c1b808451dd7'
    return hashcode


###############################################################################
# Log student entries to auto-graded questions                                #
# - modified 20240120                                                         #
###############################################################################       
def dataLogger(questionStr, x, varNames, log):
    if os.path.isfile('PHYS121_DataLogger.txt') == False:
        with open('PHYS121_DataLogger.txt', 'a+') as f:
            f.write('Date' + '\t' + 'Time' + '\t' + 'Question' + '\t' + 'Variable Name' + '\t' + 'Response' + '\t' + 'Type' + '\t' + 'Result' + '\n')
    now = datetime.now()
    corr = now - timedelta(hours = 8)

    testString = log.lower().replace('\n','').replace(' ', '').replace('name_and_student_number_1', '')
    results = []
    for k in x:
        results = results + ['passed']
    if 'failed' in testString: 
        splitList = testString.split('failed')
        for j in range(len(varNames)):
            for i in range(1, len(splitList), 2):
                if varNames[j] in splitList[i]:
                    results[j] = 'failed'
    cnt = 0
    for xi in x:
        with open('PHYS121_DataLogger.txt', 'a+') as f:
            if isinstance(xi, sym.Expr):
                objectType = 'symbolic'
            elif isinstance(xi, bool):
                objectType = 'boolean'
            elif isinstance(xi, str):
                objectType = 'string'
            elif isinstance(xi, int):
                objectType = 'integer'
            elif isinstance(xi, float):
                objectType = 'float'
            elif isinstance(xi, complex):
                objectType = 'complex'
            elif isinstance(xi, list):
                objectType = 'list'
            elif isinstance(xi, np.ndarray):
                objectType = 'numpy array'
            elif isinstance(xi, pd.DataFrame):
                objectType = 'pandas dataframe'
            elif isinstance(xi, tuple):
                objectType = 'tuple'
            elif isinstance(xi, set):
                objectType = 'set'
            elif isinstance(xi, np.ndarray) == False and isinstance(xi, list) == False and isinstance(xi, pd.DataFrame) == False and isinstance(xi, tuple) == False and isinstance(xi, set) == False and xi == ...:
                objectType = 'ellipsis'
            else:
                objectType = 'unknown'
            
            dt_string = corr.strftime("%d/%m/%Y" + '\t' + "%H:%M:%S")
            f.write(dt_string + '\t' + questionStr + '\t' + varNames[cnt] + '\t' + str(xi).replace('\n','') + '\t' + objectType + '\t' + results[cnt] + '\n')
            cnt += 1
    return

###############################################################################
# Log student entries to auto-graded questions                                #
# - modified 20240120                                                         #
###############################################################################       
def graderCheck(x, varNames, check):
    questionStr = str(check).split(' results')[0] # Get a string of the question name.
    dataLogger(questionStr, x, varNames, str(check))
    return check


###############################################################################
# For printing outputs in colour (copied from Stack Overflow)                 #
# - modified 20220607                                                         #
############################################################################### 
# Start the 'cstr' function.
def cstr2(s, color = 'black'):
    return "<b><text style=color:{}>{}</text></b>".format(color, s)

def saveMessage():
    print("\n"), display(Markdown('<img src="https://raw.githubusercontent.com/UBC-Okanagan-Physics-Labs/PHYS-121-images/main/general/gifs/save.gif" width="80">'))
    display(html_print(cstr2("STOP! Save your work before execting the next code cell.\n", color = 'red')))
    
def waitMessage():
    display(Markdown('<img src="https://raw.githubusercontent.com/UBC-Okanagan-Physics-Labs/PHYS-121-images/main/general/gifs/siren.gif" width="80">'))
    display(html_print(cstr2("Do NOT download your .zip submission yet!", color = 'red')))
    display(html_print(cstr2("Please wait for the .zip file to be created and populated with all the necessary files.", color = 'red')))
    display(html_print(cstr2("A second message will appear when it is okay to proceed.", color = 'red')))
    display(html_print(cstr2("Failure to follow these instructions may result in a corrupt submission and a grade of zero.", color = 'red')))
    display(html_print(cstr2("----------------------------------------------------------------------------", color = 'red')))
    
def proceedMessage():
    import glob
    from zipfile import ZipFile

    path = os.getcwd()
    files = glob.glob('*.zip') 

    for x in files:
        try:
            the_zip_file = ZipFile(x)
            ret = the_zip_file.testzip()
            file_list = the_zip_file.namelist()
            display(html_print(cstr2("----------------------------------------------------------------------------", color = 'blue')))
            display(Markdown('<img src="https://raw.githubusercontent.com/UBC-Okanagan-Physics-Labs/PHYS-121-images/main/general/gifs/trafficlight.gif" width="200">'))
            display(html_print(cstr2("Success!  Please proceed with the download.\n", color = 'blue')))
            print("\nThe files contained in", x, "are:\n")
            print('\n'.join(file_list))
        except:
            print("It looks like there's an issue with:", x)
            
            
            
###############################################################################
# Function for generating an animated plot of a simple pendulum               #
# - modified 20240619                                                         #
###############################################################################            
def AnimatedPlot(L, theta0, thetaList, tList, tmax):
    
    filename = 'xkcd.txt'
    lines = open(filename).read().splitlines()
    c1 = random.choice(lines)
    c2 = random.choice(lines)
    c3 = random.choice(lines)
    
    x = 0
    step = 1375
    t = tList[x]

    xBob = []
    yBob = []
    tt = []
    while t <= tmax - step*(tList[1] - tList[0]):
        xx = L*np.sin(thetaList[x])
        yy = -L*np.cos(thetaList[x])
        xBob = xBob + [xx]
        yBob = yBob + [yy]

        t = tList[x]
        tt = tt + [t]
        x = x + step

    fig = plt.figure(figsize=(15,9))
    ax1 = fig.add_subplot(2, 2, 1) 
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    
    #ax1.patch.set_facecolor('xkcd:' + c1)
    #ax1.patch.set_alpha(0.1)
    ax1.plot([0], [0], 'x', color = 'pink', markerfacecolor = "None")
    ax1.text(0.05 , 0, 'rotation axis', fontsize = 10)
    delta = 0.045
    t = ax1.text(L*np.sin(theta0) + delta , -L*np.cos(theta0) + delta, r'$\theta_0 = $' + str(round(theta0*180/np.pi)) + r'$^\circ$', fontsize = 10)
    t.set_bbox(dict(facecolor = 'white', alpha = 1, edgecolor = 'white')) # Put a solid white background behind the theta0 textbox.
    ax1.set_xlim(-L, L)
    ax1.set_ylim(-1.1*L, 0.1*L)
    ax1.set_aspect('equal')
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    
    ax2.patch.set_facecolor('xkcd:' + c2)
    ax2.patch.set_alpha(0.1)
    ax2.plot([0, tmax], [0, 0], '--', color = 'lightgrey')
    ax2.text(0.25 , -0.065, r'$y = 0$', fontsize = 10)
    ax2.plot([0, tmax], [-L, -L], '--', color = 'lightgrey')
    ax2.text(0.25 , -1.065, r'$y = -L$', fontsize = 10)
    ax2.set_xlim(0, tmax)
    ax2.set_ylim(-1.1*L, 0.1*L)
    ax2.set_xlabel('time')
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('y position', rotation = 270, labelpad = 15)
    
    ax3.patch.set_facecolor('xkcd:' + c3)
    ax3.patch.set_alpha(0.1)
    ax3.plot([0, 0], [0, tmax], '--', color = 'lightgrey')
    ax3.set_xlim(-L, L)
    ax3.set_ylim(tmax, 0)
    ax3.set_xlabel('x position')
    ax3.set_ylabel('time')

    i = 0
    t = tt[i]
    while t <= tmax - step*(tList[1] - tList[0]):
        #ax1.cla()
        if i != 0:
            ax1.plot([0, xBob[i - 1]], [0, yBob[i - 1]], 'white', linewidth = 3)
            ax1.plot([xBob[i - 1]], [yBob[i - 1]], 'o', color = 'lightgrey')
        ax1.plot(xBob[i], yBob[i], 'ko')
        ax1.plot([0, xBob[i]], [0, yBob[i]], 'k', linewidth = 1)
        
        ax2.plot(tt[i], yBob[i], 'o', color = 'blue', markerfacecolor = "None")
        ax2.plot(tt[i - 1:i + 1], yBob[i - 1:i + 1], '-', color = 'blue', alpha = 0.35, markerfacecolor = "None")

        ax3.plot(xBob[i - 1:i + 1], tt[i - 1:i + 1], '-', color = 'red', alpha = 0.35, markerfacecolor = "None")
        ax3.plot(xBob[i], tt[i], 'o', color = 'red', markerfacecolor = "None")

        display(fig)

        i = i + 1
        t = tt[i]

        clear_output(wait = True)


###############################################################################
# Simulation of a simple pendulum                                             #
# - modified 20240620                                                         #
############################################################################### 
def PendulumSim(alpha0):
    # 1. Set loop parameters (number of iterations, maximum time, and the time step).
    steps = int(1e5)
    tmax = 10 # s 
    dt = tmax/steps # Set the value of dt
    
    alpha0 = np.deg2rad(alpha0)
    
    ell = 1 # m.
    g = 9.81 # m/s^2. 

    # 4. Calculate the approximate period in units of seconds when using the small-angle approximation
    Tapprox = 2*np.pi*np.sqrt(ell/g) 

    # 2. Put the initial parameters inside lists.
    t = [0]
    alpha = [alpha0]
    v_alpha = [0]
    a_alpha = [-(g/ell)*np.sin(alpha0)]

    # 3. Set up a Python loop to iteratively calculate the quantities listed in Table 1.
    # Note that in Python, only the indented lines that follow the "for" statement are inside the loop.
    for i in range(steps): 
        # Update the time and append to the time list
        t.append(t[i] + dt)

        # Update the angular acceleration and append to the list
        a_alpha.append(-(g/ell)*np.sin(alpha[i]))

        # Update the angular velocity and append to the list
        v_alpha.append(v_alpha[i] + a_alpha[i]*dt)

        # Update the angular position and append to the list
        alpha.append(alpha[i] + v_alpha[i]*dt)

    # 4. Plot the angular position as a function of time
    plt.plot(t, np.rad2deg(alpha))
    plt.plot([0, tmax], [0, 0], ':', color = 'lightgrey')
    plt.xlabel('time (s)')
    plt.ylabel('angular position (degrees)')
    plt.xlim(0, tmax);

    # 5. Determine the oscillation period.
    i = 0
    while not (v_alpha[i] <= 0 and v_alpha[i + 1] > 0):
        i = i + 1

    # 6. Compare the numerically-calculated period to the period found using the small-angle approximation.
    T = 2*t[i]
    display(Latex(r'$T = $ ' + '{0:.3f}'.format(T) + ' s'))
    display(Latex(r'$T_\mathrm{approx} = $ ' + '{0:.3f}'.format(Tapprox) + ' s'))




###############################################################################
# Weighted & Unweighted Lorentzian Fits                                       #
# - modified 20221014                                                         #
############################################################################### 
# Start the 'Lorentz' function.
def Lorentz(xData, yData, yErrors = [], start = [1, 220, 50], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    A_fit = ''
    w0_fit = ''
    gamma_fit = ''
    errA = ''
    errw0 = ''
    errgamma = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    if  type(start).__module__ == np.__name__:
        start = start.tolist()
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(start) != 3:
        display(html_print(cstr('The length of start (' + str(len(start)) + ') must be equal to 3.', color = 'magenta')))
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in start) != True:
        display(html_print(cstr("The elements of 'start' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def LorentzFcn(x, A, w0, gamma):
            y = A/np.sqrt(1 + (x/gamma)**2*(1 - (w0/x)**2)**2)
            return y
        
        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        display(Markdown(r'$y = \dfrac{A}{\sqrt{1 + \left(\dfrac{\omega}{\gamma}\right)^2\left[1 - \left(\dfrac{\omega_0}{\omega}\right)^2\right]^2}}$'))
    
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(LorentzFcn, xData, yData, p0 = start)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(LorentzFcn, xData, yData, sigma = yErrors, p0 = start, absolute_sigma = True)
            display(Markdown('This is a **WEIGHTED** fit.'))

        A_fit = a_fit[0]
        errA = np.sqrt(np.diag(cov))[0]
        w0_fit = a_fit[1]
        errw0 = np.sqrt(np.diag(cov))[1]
        gamma_fit = a_fit[2]
        errgamma = np.sqrt(np.diag(cov))[2]
        
        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        A = uncertainties.ufloat(A_fit, errA)
        w0 = uncertainties.ufloat(w0_fit, errw0)
        gamma = uncertainties.ufloat(gamma_fit, errgamma)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
#            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits + '/' + xUnits + eval(r'"\u00b' + str(Power) + '"')},
#                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
#                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}
            my_dict = {'amplitude' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = my_dict = {'amplitude' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        elif xUnits == '' and yUnits != '':
            my_dict = my_dict = {'amplitude' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits},
                       'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        else:
            my_dict = my_dict = {'amplitude' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A)},
                       'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0)},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma)}}

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        #fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)

        plt.plot(xx, A_fit/np.sqrt(1 + (xx/gamma_fit)**2*(1 - (w0_fit/xx)**2)**2), 'k-')

        # Show the final plot.
        plt.show()
    return A_fit, w0_fit, gamma_fit, errA, errw0, errgamma, fig


###############################################################################
# Weighted & Unweighted Lorentzian Phase Fits                                 #
# - modified 20221014                                                         #
############################################################################### 
# Start the 'Lorentz' function.
def Phase(xData, yData, yErrors = [], start = [220, 50], xlabel = 'x-axis', ylabel = 'y-axis', xUnits = '', yUnits = ''):
    # Check to see if the elements of dataArray are numpy arrays.  If they are, convert to lists
    w0_fit = ''
    gamma_fit = ''
    errw0 = ''
    errgamma = ''
    fig = ''
    if  type(xData).__module__ == np.__name__:
        xData = xData.tolist()
    if  type(yData).__module__ == np.__name__:
        yData = yData.tolist()
    if  type(yErrors).__module__ == np.__name__:
        yErrors = yErrors.tolist()
    if  type(start).__module__ == np.__name__:
        start = start.tolist()   
    # Check that the lengths of the inputs are all the same.  Check that the other inputs are strings.
    if len(start) != 2:
        display(html_print(cstr('The length of start (' + str(len(start)) + ') must be equal to 2', color = 'magenta')))
    if len(xData) != len(yData):
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yData (' + str(len(yData)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(xData) != len(yErrors):  
        display(html_print(cstr('The length of xData (' + str(len(xData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif len(yErrors) != 0 and len(yData) != len(yErrors):  
        display(html_print(cstr('The length of yData (' + str(len(yData)) + ') is not equal to the length of yErrors (' + str(len(yErrors)) + ').', color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in xData) != True:
        display(html_print(cstr("The elements of 'xData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in yData) != True:
        display(html_print(cstr("The elements of 'yData' must be integers or floats.", color = 'magenta')))
    elif all(isinstance(x, (int, float)) for x in start) != True:
        display(html_print(cstr("The elements of 'start' must be integers or floats.", color = 'magenta')))
    elif len(yErrors) != 0 and all(isinstance(x, (int, float)) for x in yErrors) != True:
        display(html_print(cstr("The elements of 'yErrors' must be integers or floats.", color = 'magenta')))
    elif isinstance(xlabel, str) == False:
        display(html_print(cstr("'xlabel' must be a string.", color = 'magenta')))
    elif isinstance(ylabel, str) == False:
        display(html_print(cstr("'ylabel' must be a string.", color = 'magenta')))
    elif isinstance(xUnits, str) == False:
        display(html_print(cstr("'xUnits' must be a string.", color = 'magenta')))
    elif isinstance(yUnits, str) == False:
        display(html_print(cstr("'yUnits' must be a string.", color = 'magenta')))
    else:
        # Uncertainties is a nice package that can be used to properly round
        # a numerical value based on its associated uncertainty.
        install_and_import('uncertainties') # check to see if uncertainties is installed.  If it isn't attempt to do the install
        import uncertainties

        # Define the linear function used for the fit.
        def LorentzFcn(x, w0, gamma):
            y = np.arctan((x/gamma)*(1 - (w0/x)**2))
            return y
        
        # If the yErrors list is empty, do an unweighted fit.  Otherwise, do a weighted fit.
        print('')
        display(Markdown(r'$y = \arctan\left\{\dfrac{\omega}{\gamma}\left[1 - \left(\dfrac{\omega_0}{\omega}\right)^2\right]\right\}$'))
    
        if len(yErrors) == 0: 
            a_fit, cov = curve_fit(LorentzFcn, xData, yData, p0 = start)
            display(Markdown('This is an **UNWEIGHTED** fit.'))
        else:
            a_fit, cov = curve_fit(LorentzFcn, xData, yData, sigma = yErrors, p0 = start, absolute_sigma = True)
            display(Markdown('This is a **WEIGHTED** fit.'))

        w0_fit = a_fit[0]
        errw0 = np.sqrt(np.diag(cov))[0]
        gamma_fit = a_fit[1]
        errgamma = np.sqrt(np.diag(cov))[1]
        
        # Use the 'uncertainties' package to format the best-fit parameters and the corresponding uncertainties.
        w0 = uncertainties.ufloat(w0_fit, errw0)
        gamma = uncertainties.ufloat(gamma_fit, errgamma)

        # Make a formatted table that reports the best-fit parameters and their uncertainties        
        import pandas as pd
        if xUnits != '' and yUnits != '':
#            my_dict = {'coefficient' :{'':'$A =$', 'Value': '{:0.2ug}'.format(A), 'Units': yUnits + '/' + xUnits + eval(r'"\u00b' + str(Power) + '"')},
#                       'power':{'':'$N =$', 'Value': '{:0.2ug}'.format(N), 'Units': ''},
#                       'offset':{'':'$C =$', 'Value': '{:0.2ug}'.format(C), 'Units': yUnits}}
            my_dict = {'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        elif xUnits != '' and yUnits == '':
            my_dict = my_dict = {'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        elif xUnits == '' and yUnits != '':
            my_dict = my_dict = {'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0), 'Units': xUnits},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma), 'Units': xUnits}}
        else:
            my_dict = my_dict = {'angular resonance freuqnecy' :{'':'$\omega_0 =$', 'Value': '{:0.2ug}'.format(w0)},
                       'width' :{'':'$\gamma =$', 'Value': '{:0.2ug}'.format(gamma)}}

        # Display the table
        df = pd.DataFrame(my_dict)
        display(df.transpose())
        
        # Generate the best-fit line. 
        #fitFcn = np.polynomial.Polynomial(a_fit)
        
        # Call the Scatter function to create a scatter plot.
        fig = Scatter(xData, yData, yErrors, xlabel, ylabel, xUnits, yUnits, False, False)
        
        # Determine the x-range.  Used to determine the x-values needed to produce the best-fit line.
        if np.min(xData) > 0:
            xmin = 0.9*np.min(xData)
        else:
            xmin = 1.1*np.min(xData)
        if np.max(xData) > 0:
            xmax = 1.1*np.max(xData)
        else:
            xmax = 0.9*np.max(xData)

        # Plot the best-fit line...
        xx = np.arange(xmin, xmax, (xmax-xmin)/5000)

        plt.plot(xx, np.arctan((xx/gamma_fit)*(1 - (w0_fit/xx)**2)), 'k-')

        # Show the final plot.
        plt.show()
    return w0_fit, gamma_fit, errw0, errgamma, fig





###############################################################################
# Report the time since last notebook save                                    #                                
# - modified 20260115                                                         #
############################################################################### 
def save_time():
    import time, pathlib, datetime

    cwd = pathlib.Path().resolve()
    ipynbs = list(cwd.glob("*.ipynb"))

    lines = []
    lines.append(f"Kernel working directory: {cwd}")
    lines.append(
        f"Log generated at: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if not ipynbs:
        lines.append("⚠️ No notebook file found in this folder.")
    else:
        latest = max(ipynbs, key=lambda p: p.stat().st_mtime)
        age = time.time() - latest.stat().st_mtime
        mtime = datetime.datetime.fromtimestamp(
            latest.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        lines.append(f"🕒 Time since last notebook save: {age:.1f} seconds")
        lines.append(f"📓 Most recently saved notebook: {latest.name}")
        lines.append(f"📅 That notebook's last-modified time: {mtime}")

        if age > 10:
            lines.append("")
            lines.append(
                "⚠️ If you made any changes since the last save, "
                "please save the notebook (Ctrl/Cmd-S) and then rerun this export cell."
            )

    text = "\n".join(lines) + "\n"

    # Print for the student
    print(text)

    # Write for grading/auditing
    (cwd / "save_time.txt").write_text(text, encoding="utf-8")
