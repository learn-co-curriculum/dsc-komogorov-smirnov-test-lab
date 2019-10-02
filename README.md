
# The Kolmogorov-Smirnov Test - Lab

## Introduction
In the previous lesson, we saw that the Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution, or between the empirical distribution functions of two samples. In this lab, we shall see how to perform this test in python. 

## Objectives

You will be able to:
* Perform 1 sample and 2 sample KS tests in Python and Scipy
* Compare the KS test to visual approaches for checking normality assumptions
* Plot the CDF and ECDF to visualize parametric and empirical cumulative distribution functions

## Generate Data

### Let's import the necessary libraries and generate some data 


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


```python
# __SOLUTION__ 
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



![png](index_files/index_6_1.png)



![png](index_files/index_6_2.png)


    x_50



![png](index_files/index_6_4.png)



![png](index_files/index_6_5.png)


    x_100



![png](index_files/index_6_7.png)



![png](index_files/index_6_8.png)


    x_1000



![png](index_files/index_6_10.png)



![png](index_files/index_6_11.png)



```python
# __SOLUTION__ 
# Plot histograms and QQplots for above datasets

# You code here

labels = ['x_10','x_50','x_100','x_1000']
for ind, i in enumerate([x_10,x_50,x_100,x_1000]):
    print (labels[ind])
    plt.hist(i)
    sm.qqplot(i, line='s')
    plt.show()
```

    x_10



![png](index_files/index_7_1.png)



![png](index_files/index_7_2.png)


    x_50



![png](index_files/index_7_4.png)



![png](index_files/index_7_5.png)


    x_100



![png](index_files/index_7_7.png)



![png](index_files/index_7_8.png)


    x_1000



![png](index_files/index_7_10.png)



![png](index_files/index_7_11.png)



```python
# You comments here 
```


```python
# __SOLUTION__ 
# You comments here 
```


```python
# __SOLUTION__ 
# Histograms should not be used solely to detect normality directly
# Histograms are better to look for symmetry, skewness, and outliers 
# These can instead be used to get an indications of non-normality. 

# We see some outliers in our datasets
# no clear indications of non-normality for each plot.
```


```python
# __SOLUTION__ 
# The QQ plot is a much better visualization of data as gives a reference to compare against  
# Shows a better picture about normality instead of relying on the histograms (or box plots).
# From QQ plot we can be more assured our data is normal - compared to non normality check in histogram
```

### Create a function to plot the normal CDF and ECDF for a given dataset
- Create a function ks_plot(data) to generate an empirical CDF from data
- Create a normal CDF using the same mean = 0 and sd = 3, having the same number of values as data


```python
# You code here 

def ks_plot(data):

    pass
    
# Uncomment below to run the test
# ks_plot(stats.norm.rvs(loc=0, scale=3, size=100)) 
# ks_plot(stats.norm.rvs(loc=5, scale=4, size=100))

```


![png](index_files/index_13_0.png)



![png](index_files/index_13_1.png)



```python
# __SOLUTION__ 
# You code here 

def ks_plot(data):

    plt.figure(figsize=(10, 7))
    plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))
    plt.plot(np.sort(stats.norm.rvs(loc=0, scale=3, size=len(data))), np.linspace(0, 1, len(data), endpoint=False))

    plt.legend(['ECDF', 'CDF'])
    plt.title('Comparing CDFs for KS-Test, Sample size=' + str(len(data)))
    

ks_plot(stats.norm.rvs(loc=0, scale=3, size=100)) 
ks_plot(stats.norm.rvs(loc=5, scale=4, size=100))

```


![png](index_files/index_14_0.png)



![png](index_files/index_14_1.png)


This is awesome. The difference between the two CDFs in the second plot shows that the sample did not come from the distribution which we tried to compare it against. 

### Now you can run all the generated datasets through the function ks_plot and comment on the output.


```python
# Your code here 
```


![png](index_files/index_16_0.png)



![png](index_files/index_16_1.png)



![png](index_files/index_16_2.png)



![png](index_files/index_16_3.png)



```python
# __SOLUTION__ 
ks_plot(x_10)
ks_plot(x_50)
ks_plot(x_100)
ks_plot(x_1000)
```


![png](index_files/index_17_0.png)



![png](index_files/index_17_1.png)



![png](index_files/index_17_2.png)



![png](index_files/index_17_3.png)



```python
# Your comments here 

```


```python
# __SOLUTION__ 
# Your comments here 

# As we have more data values to compare, we get a better idea of normality
# Due to randomness in smaller sample sizes, it is very likely that the value of d would be high
# As our sample size goes from 50 to a 1000, we are in a much better position to comment on normality
```

### KS test in SciPy

Let's run the Kolmogorov-Smirnov test, and use some statistics to get a final verdict on normality. It lets us test the hypothesis that the sample is a part of the standard t-distribution. In SciPy, we run this test using the method below:

```python
scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
```
Details on arguments being passed in can be viewed at this [link to the official doc.](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)


### Run the KS test for normality assumption using the datasets created earlier and comment on the output
- Perform the KS test against a normal distribution with mean = 0 and sd = 3
- If p < .05 we can reject the null hypothesis and conclude our sample distribution is not identical to a normal distribution.


```python
# Perform KS test 

# Your code here 

# KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
# KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
# KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
# KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)
```

    KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
    KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
    KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
    KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)



```python
# __SOLUTION__ 
# Perform KS test 

# Your code here 
np.random.seed(999)
for i in [x_10,x_50,x_100,x_1000]:
    print (stats.kstest(i, 'norm', args=(0, 3)))


# KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
# KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
# KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
# KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)
```

    KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
    KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
    KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
    KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)



```python
# Your comments here 

```


```python
# __SOLUTION__ 
# Your comments here 

# The P-value in all cases is much greater than .05 
# We cannot reject the Null Hypothesis i.e. our sample is IDENTICAL to a normal distribution
# This is very intuitive as we started off with normal distributions
```


### Generate a uniform distribution and plot / calculate the ks test against a uniform as well as a normal distribution


```python
# Try with a uniform distribution
x_uni = np.random.rand(1000)

# KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
# KstestResult(statistic=0.5000553288071681, pvalue=0.0)
```

    KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
    KstestResult(statistic=0.5000553288071681, pvalue=0.0)



```python
# __SOLUTION__ 
# Try with a uniform distubtion
x_uni = np.random.rand(1000)
print(stats.kstest(x_uni, lambda x: x))
print(stats.kstest(x_uni, 'norm', args=(0, 3)))

# KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
# KstestResult(statistic=0.5000553288071681, pvalue=0.0)
```

    KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
    KstestResult(statistic=0.5000553288071681, pvalue=0.0)



```python
# Your comments here 

```


```python
# __SOLUTION__ 
# Your comments here 

# In the first case, the p-value is much larger than 0.05 so we cannot reject the Null Hypothesis 
# and conclude that our sample is a uniform distribution
# In the second case, comparing a uniform distribution against a normal CDF, the p value - 0 
# so we reject the Null Hypothesis with a high degree of confidence 
```

## 2 sample KS test
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


![png](index_files/index_33_0.png)



```python
# __SOLUTION__ 
# Generate binomial data
N = 1000
x_1000_bi = np.concatenate((np.random.normal(-1, 1, int(0.1 * N)), np.random.normal(5, 1, int(0.4 * N))))[:, np.newaxis]
plt.hist(x_1000_bi);
```


![png](index_files/index_34_0.png)


### Plot the CDFs for x_1000_bimodal and x_1000 and comment on the output 


```python

# Plot the CDFs
def ks_plot_2sample(data_1, data_2):
    '''
    Data entereted must be the same size.
    '''
    pass

# Uncomment below to run
# ks_plot_2sample(x_1000, x_1000_bi[:,0])

```


![png](index_files/index_36_0.png)



```python
# __SOLUTION__ 

# Plot the CDFs
def ks_plot_2sample(data_1, data_2):
    '''
    Data entereted must be the same size.
    '''
    length = len(data_1)
    plt.figure(figsize=(12, 7))
    plt.plot(np.sort(data_1), np.linspace(0, 1, len(data_1), endpoint=False))
    plt.plot(np.sort(data_2), np.linspace(0, 1, len(data_2), endpoint=False))
    plt.legend('top right')
    plt.legend(['Data_1', 'Data_2'])
    plt.title('Comparing 2 CDFs for KS-Test')
    
ks_plot_2sample(x_1000, x_1000_bi[:,0])
```


![png](index_files/index_37_0.png)



```python
# You comments here 

```


```python
# __SOLUTION__ 
# You comments here 

# x_1000 and x_1000_bi diverge a lot 
# We can expect a high value for the d statistic 

```

### Run the two-sample KS test on x_1000 and x_1000_bi and comment on the results


```python
# Your code here

# Ks_2sampResult(statistic=0.633, pvalue=4.814801487740621e-118)
```


```python
# __SOLUTION__ 
# Check if the distributions are equal
stats.ks_2samp(x_1000, x_1000_bi[:,0])

# Ks_2sampResult(statistic=0.633, pvalue=4.814801487740621e-118)
```




    Ks_2sampResult(statistic=0.633, pvalue=4.814801487740621e-118)




```python
# Your comments here 


```


```python
# __SOLUTION__ 
# Your comments here 

# A very small p-value , hence we reject the Null hypothesis
# The two samples belog to different distributions
```

## Summary

In this lesson, we saw how to check for normality (and other distributions) using one sample and two sample ks-tests. You are encouraged to use this test for all the upcoming algorithms and techniques that require a normality assumption. We saw that we can actually make assumptions for different distributions by providing the correct CDF function into Scipy KS test functions. 
