
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class Screening():
    def __init__(self, df):
        self.df = df

    def analyze_col(self,col,plot=False,prints=False, continuous=False):
        copy = self.df.loc[:,['SalePrice', col]]
        #Drop nulls
        copy = copy.replace('None or Unspecified', np.NaN)
        if continuous==True:
            copy = copy.replace(0, np.NaN)
        copy = copy.dropna(axis=0)
        test = copy.loc[:,['SalePrice',col]]
        test = test.dropna(axis=0).reset_index(drop=True)
        
        if continuous == False:
            #Run F-test
            f,pval = f_oneway(*[list(test['SalePrice'][test[col] == name].values) for name in set(test[col])])
            #print
            if prints == True:
                print('Ftest pval: ' + str(pval))
                print('Len: ' + str(test.count()))
                print('Unique Values: ' + str(copy[col].value_counts()))
            #Plot
            if plot == True:
                fig, ax = plt.subplots(1,1, figsize=(5,5))
                sns.boxplot(test[col], test['SalePrice'],  ax = ax)
                plt.xlabel(col)
                plt.tight_layout()
            return pval
        
        if continuous == True:
            
            #Run R^2 
            r2 = test['SalePrice'].corr(test[col])
            if prints == True:
                print('R-square: ' + str(r2))
                print('Len: ' + str(test.SalePrice.count()))

            if plot == True:
                plt.scatter(test[col], test['SalePrice'])
                plt.xlabel(col)
                plt.ylabel('Sale Price')
            return r2
        else:
            raise ValueError('Continuous true or false only.')
        
    def analyze_Rsquare(self,col,plot=False,prints=False, continuous=False):
        copy = self.df.loc[:,['SalePrice', col]]
        #Drop nulls
        copy = copy.replace('None or Unspecified', np.NaN)
        #if continuous==True:
            #copy = copy.replace(0, np.NaN)
        copy = copy.dropna(axis=0)
        test = copy.loc[:,['SalePrice',col]]
        test = test.dropna(axis=0).reset_index(drop=True)
        
        if continuous == False:
    
            rsquares = []
            if len(test[col].value_counts()) <= 2:

                r2 = test['SalePrice'].corr(test.loc[:,col])
                rsquares.append([col, r2, test.iloc[:,i+1].count()])
            else:
                test = pd.get_dummies(test, prefix=None, prefix_sep='_', 
                dummy_na=False, columns=None, sparse=False, 
                drop_first=False, dtype=None)
                length = len(test.columns)-1
                for i in range(length):
                    r2 = test['SalePrice'].corr(test.iloc[:,i+1])
                    rsquares.append([test.columns[i+1], r2, test.iloc[:,i+1].count()])
            #print
            if prints == True:
                print('Rsquares: ' + str(rquares))
                print('Len: ' + str(test.count()))
                #Print column names
                print(test.columns)
            #Plot
            if plot == True:
                fig, ax = plt.subplots(1,1, figsize=(5,5))
                #test[col].hist(ax=ax1)
                sns.boxplot(test[col], test['SalePrice'],  ax = ax)
                plt.xlabel(col)
                #plt.ylabel('Sale Price')
                plt.tight_layout()
            return rsquares

        if continuous == True:
            
            #Run R^2 
            r2 = test['SalePrice'].corr(test[col])
            if prints == True:
                print('R-square: ' + str(r2))
                print('Len: ' + str(test.SalePrice.count()))

            if plot == True:
                plt.scatter(test[col], test['SalePrice'])
                plt.xlabel(col)
                plt.ylabel('Sale Price')
            return r2
        else:
            raise ValueError('Continuous true or false only.')


class DataType(BaseEstimator, TransformerMixin):
    """Cast the data types of the id and data source columns to strings
    from numerics and convert 'saledate' to dt
    """
    
    def fit(self, X, y):
        return self

    def transform(self, X): 
        id_cols = ['SalesID', 'MachineID', 'ModelID']
        X[id_cols] = X[id_cols].astype('str', inplace=True)
        X['saledate'] = pd.to_datetime(X.saledate)
        return X


class FilterColumns(BaseEstimator, TransformerMixin):
    """Only keep columns that don't have NaNs.
    """
    def fit(self, X, y):
        column_counts = X.count(axis=0)
        self.keep_columns = column_counts[column_counts == column_counts.max()]
        return self

    def transform(self, X):
        return X.loc[:, self.keep_columns.index]


class ReplaceOutliers(BaseEstimator, TransformerMixin):
    """Replace year made when listed as earlier than 1900, with
    mode of years after 1900. Also add imputation indicator column.
    """
    def fit(self, X, y):
        self.replace_value = X.YearMade[X.YearMade > 1900].mode()
        return self

    def transform(self, X):
        condition = X.YearMade > 1900
        X['YearMade_imputed'] = 0
        X.loc[~condition, 'YearMade'] = self.replace_value[0]
        X.loc[~condition, 'YearMade_imputed'] = 1
        return X


class ComputeAge(BaseEstimator, TransformerMixin):
    """Compute the age of the vehicle at sale.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['equipment_age'] = X.saledate.dt.year - X.YearMade
        return X


class ComputeNearestMean(BaseEstimator, TransformerMixin):
    """Compute a mean price for similar vehicles.
    """
    def __init__(self, window=5):
        self.window = window

    def get_params(self, **kwargs):
        return {'window': self.window}

    def fit(self, X, y):
        X = X.sort_values(by=['saledate'])
        g = X.groupby('ModelID')['SalePrice']
        m = g.apply(lambda x: x.rolling(self.window).agg([np.mean]))
        
        ids = X[['saledate', 'ModelID', 'SalesID']]
        z = pd.concat([m, ids], axis=1)
        z['saledate'] = z.saledate + timedelta(1)

        # Some days have more than 1 transaction for a particular model,
        # use the last mean
        z = z.drop('SalesID', axis=1)
        groups = ['ModelID', 'saledate']
        self.averages = z.groupby(groups).apply(lambda x: x.tail(1))
        self.averages = self.averages.drop(groups, axis=1)

        self.default_mean = X.SalePrice.mean()
        return self

    def transform(self, X):
        near_price = pd.merge(self.averages, X, how='outer',
                              on=['ModelID', 'saledate'])
        nxcols = ['ModelID', 'saledate']
        near_price = near_price.set_index(nxcols).sort_index()
        g = near_price['mean'].groupby(level=0)
        #forward fill empty rows
        filled_means = g.transform(lambda x: x.fillna(method='ffill'))
        near_price['filled_mean_price'] = filled_means
        #drop null rows
        near_price = near_price[near_price['SalesID'].notnull()]
        #create an imputation for missing means
        missing_mean = near_price.filled_mean_price.isnull()
        near_price['no_recent_transactions'] = missing_mean
        #fillna with overall saleprice mean
        near_price['filled_mean_price'].fillna(self.default_mean, inplace=True)
        return near_price


class ColumnFilter(BaseEstimator, TransformerMixin):
    """Only use the following columns.
    """
    columns = ['YearMade', 'YearMade_imputed', 'equipment_age',
               'filled_mean_price', 'no_recent_transactions']

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        X = X.set_index('SalesID')[self.columns].sort_index()
        return X


def rmsle(y, y_hat):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    """
    assert len(y) == len(y_hat)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_hat), 2)))


if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    df = df.set_index('SalesID').sort_index()
    y = df.SalePrice
    print(any(y < 0))
    # This is for predefined split... we want -1 for our training split,
    # 0 for the test split.
    cv_cutoff_date = pd.to_datetime('2011-01-01')
    cv = -1*(pd.to_datetime(df.saledate) < cv_cutoff_date).astype(int)

    cross_val = PredefinedSplit(cv)

    p = Pipeline([
        ('filter', FilterColumns()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('lm', LinearRegression())
    ])
    df = df.reset_index()

    # GridSearch
    params = {'nearest_average__window': [3, 5, 7]}

    # Turns rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, params,
                        cv=cross_val,
                        n_jobs=-1)
    clf = gscv.fit(df.reset_index(drop=True), y)
    

    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(-1*clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)
