
# The Kolmogorov-Smirnov Test - Lab

## Introduction
In the previous lesson, we saw that the Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution, or between the empirical distribution functions of two samples. In this lab, we shall see how to perform this test in python. 

## Objectives

You will be able to:
* Perform 1 sample and 2 sample KS tests in Python and Scipy
* Compare KS test to visual approaches for checking normality assumptions
* Plot CDF and ECDF to visualize parametric and empirical cumulative distribution functions

## Generate Data

### Let's import necessary libraries and generate some data 


```python
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Create the normal random variables with mean 0, and sd 3
x_10 = stats.norm.rvs(loc=0, scale=3, size=10)
x_50 = stats.norm.rvs(loc=0, scale=3, size=50)
x_100 = stats.norm.rvs(loc=0, scale=3, size=100)
x_1000 = stats.norm.rvs(loc=0, scale=3, size=1000)
```

### Plot Histograms and QQ plots of above datasets and comment on the output 

- How good are these techniques for checking normality assumptions?
- Compare both these techniques and identify their limitations/benefits etc. 



```python
# Plot histograms and QQplots for above datasets

# You code here

```

    x_10



![png](index_files/index_5_1.png)



![png](index_files/index_5_2.png)


    x_50



![png](index_files/index_5_4.png)



![png](index_files/index_5_5.png)


    x_100



![png](index_files/index_5_7.png)



![png](index_files/index_5_8.png)


    x_1000



![png](index_files/index_5_10.png)



![png](index_files/index_5_11.png)



```python
# You comments here 
```

### Creat a function to plot the normal CDF and ECDF for a given dataset
- Create a function ks_plot(data) to generate an empirical CDF from data
- Create a normal CDF using the same mean = 0 and sd = 3 , having same number of values as data


```python
# You code here 

def ks_plot(data):

    pass
    
# Uncomment below to run the test
# ks_plot(stats.norm.rvs(loc=0, scale=3, size=100)) 
# ks_plot(stats.norm.rvs(loc=5, scale=4, size=100))


```


![png](index_files/index_8_0.png)



![png](index_files/index_8_1.png)


This is awesome. The difference between two cdfs in the second plot show that sample did not come from the distribution which we tried to compare it against. 

### Now you can run all the generated datasets through the function ks_plot and comment on the output.


```python
# Your code here 
```


![png](index_files/index_10_0.png)



![png](index_files/index_10_1.png)



![png](index_files/index_10_2.png)



![png](index_files/index_10_3.png)



```python
# Your comments here 

```

### KS test in SciPy

Lets run the Kolmogorov-Smirnov test, and use some statistics to get a final verdict on normality. It lets us test the hypothesis that the sample is a part of the standard t-distribution. In SciPy, we run this test using the method below:

```python
scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
```
Details on arguments being passed in can be viewed at this [link to official doc.](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)


### Run KS test for normality assumption using the datasets created earlier and comment on the output
- Perform test KS test against a normal distribution with mean = 0 and sd = 3
- If p < .05 we can reject the null, and conclude our sample distribution is not identical to a normal distribution.


```python
# Perform KS test 

# Your code here 

for i in [x_10,x_50,x_100,x_1000]:
    print (stats.kstest(i, 'norm', args=(0, 3)))


# KstestResult(statistic=0.20726402525186666, pvalue=0.7453592647579976)
# KstestResult(statistic=0.11401670469341446, pvalue=0.506142501491317)
# KstestResult(statistic=0.06541325864884379, pvalue=0.7855843705750273)
# KstestResult(statistic=0.026211483799585156, pvalue=0.4974218016349998)
```

    KstestResult(statistic=0.1632179747086434, pvalue=0.9526730195059537)
    KstestResult(statistic=0.1072471715631031, pvalue=0.590698827561229)
    KstestResult(statistic=0.08957748590021086, pvalue=0.3787922342822605)
    KstestResult(statistic=0.02825842101747439, pvalue=0.39744474314954137)



```python
# Your comments here 

```


### Generate a uniform distribution and plot / calculate the ks test against a uniform as well as a normal distribution


```python
# Try with a uniform distubtion
x_uni = np.random.rand(1000)


# KstestResult(statistic=0.025244449633212818, pvalue=0.5469114859681035)
# KstestResult(statistic=0.5001459915784039, pvalue=0.0)
```

    KstestResult(statistic=0.025244449633212818, pvalue=0.5469114859681035)
    KstestResult(statistic=0.5001459915784039, pvalue=0.0)



```python
# Your comments here 

```

## 2 - sample KS test
A two sample KS test is available in SciPy using following function
```python 
scipy.stats.ks_2samp(data1, data2)[source]
```

Let's generate some bi-modal data first for this test 


```python
# Generate binomial data
N = 1000
x_1000_bi = np.concatenate((np.random.normal(-1, 1, int(0.1 * N)), np.random.normal(5, 1, int(0.4 * N))))[:, np.newaxis]
plt.hist(x_1000_bi);
```


![png](index_files/index_21_0.png)


### Plot the CDFs for x_100_bimodal and x_1000 and comment on the output 


```python

# Plot the CDFs
def ks_plot_2sample(data_1, data_2):
    '''
    Data entereted must be the same size.
    '''
    pass

# Uncomment below to run
# ks_plot_comp(x_100, x_bimodal_100[:,0])
```


![png](index_files/index_23_0.png)



```python
# You comments here 

```

### Run the two sample KS test on x_100 and x_100_bi and comment on the results


```python
# You rcode here

# Ks_2sampResult(statistic=0.575, pvalue=1.2073337530608254e-14)
```


```python
# Your comments here 


```

## Summary

In this lesson, we saw how to check for normality (and other distributions) using one sample and two sample ks-tests. You are encouraged to use this test for all the upcoming algorithms and techniques that require a normality assumption. We saw that we can actually make assumptions for different distributions by providing the correct CDF function into Scipy KS test functions. 
