## Capstone Inferential Statistics

For this portion of the project, I will use some of the inferential statistics techniques I've learned to gain greater insight into my cleaned data set.  The final goal for the capstone project will be to predict rent prices in the DC area by using a regression model.  One of the models we will try to use, is linear regression. Although normality of the dependent variables is not a requirement for linear regression, it would behoove us to check the distribution of rent prices to see if they are normally distributed.  

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Regular%20distplot.png)

As noted in the Data Story, the distribution of rent prices seems to be right skewed. This may suggest that there are some extreme values pulling the mean of the distribution up and the median will be a better statistic for understanding the middle values.  Perhaps, taking the log of the rent prices will provide us with a normally distributed plot.

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Logfit%20displot.png)

As we can see from the plot above, the distribution appears to look more normal after the log transformation. Included on the plot, in black, is a theoretical normal distribution for the values we have.  The distplot appears to be pretty close to this normal projection.  However, for interpretation purposes, it is important to note that if we were to create a regression model with rent price(y) and indepdent variables(x), the betas of the independent variables would be the percentage increase in y as a result of a one unit increase in x.  

What are the most important independent variables?

In order to answer this question, we will create a correlation matrix with all of the independent variables and dependent variable to see which 9 have the highest correlation with rent price in order to gain greater insight into what effects the price.

**Highest Positive Correlation**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Positive%20Corr.png)


**Highest Negative Correlation**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Negative%20Corr.png)

As we can see from the tables above, the top three most influential features according to the correlation matrix are sqft, bathrooms, and bedrooms (in that order).  The next closest feature has half the correlation coefficient of bedrooms ('Cooktop').  It's also interesing to note that the negatively correlated features have fairly weak correlations with the highest being around -0.12.  Perhaps even more interesting, the classic trope, Location! Location! Location! seems not to be as important as the top three features. Northwest Washington is the highest correlated locality with price and it's correlation coefficient is only 0.203816.  Because of this, we will take a closer look into the top three correlated with price. Lastly, it is always important to note whenever dealing with correlations that correlation != causation. 

**Top Three Correlations**

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20SQFT.png)

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20Baths.png)

![Image](https://github.com/TCummings03/Springboard-Capstone-Inferential-Statistics/blob/master/Images/Rent%20vs%20Beds.png)

From the joint plots above, we can see a couple of different things. 1) A histogram of each variable 2) the pearsonr & p value 3) regression line.  All three have positive correlations with price, which suggests the more sqft, beds, or baths, the higher the rent price. This makes sense because the larger the apartment, the higher the price. What's interesting to note, however, about the sqft distribution is that it appears to be suffering from heteroskedasticity, which occurs when the variability of a variable is unequal acorss the range of values of a second variable that predicts it.  Due to the hetereoskedasticity, it may be wise to also take a log transformation of sqft when running our regression model. The distribution of bedrooms and bathrooms seems to be right skewed, which may suggest that they too could benefit from log transformations.  Although this is not a requirement for linear regression, it may help improve our model and make interpretation of coefficients easier. As far as the central limit is concerned, we can be comfortable with satisfying the benchmark of more than 30 observations since our dataset consists of nearly 12,000 listings. 

**Base Case**

Figuring out the best loss function to score our model is crucial. Whether you're using RMSE, MSE, r^2, etc. it is important to define your scoring fucntion.  Another critical inferential statistic skill that is necessary for analyze our results is creating a "baseline case." This baseline case will help serve as the backdrop against which we can compare the results of our model.  We will use this baseline case in conjunction with our scoring function of Root Mean Squared Error to evaluate our model.  The base case is found by taking the median price/ median sqft and multiplying it by each respective listing's sqft. This "base case" is a fairly crude way of predicting rent price. However, this is useful as a bench mark to see if using a model will actually be useful in trying to predict rent price.  At a minimum, the model should do better than the RMSE from the base case (1101.4502853). Here is the code for the base case:

```#Get a baseline case to compare model against:

median_price = df1.rent_price.median()

median_sqft = df1.sqft.median()

median_p_sqft = median_price/median_sqft

baselinep = [median_p_sqft * x for x in df1.sqft]

err = df1.sqft - baselinep

sq_err = err ** 2

mean_sq_err = np.mean(sq_err)

root_mse = np.sqrt(mean_sq_err)

print('Median Rent Price:', df1.rent_price.median())
print('Mean Rent Price:', df1.rent_price.mean())
print('RMSE Base Case:', root_mse)

Median Rent Price: 2100.0
Mean Rent Price: 2365.5429759257468
RMSE Base Case: 1101.4502853
```


Sources:
http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html
