import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from error import ErrorHandler

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns

	def fit(self, xs, ys, **params):
		return self

	def transform(self, xs):
		return xs[self.columns]


def score(df, df_x, pipe, grid):
	n = input('\nChoose scoring:\n1. r^2\n2. MAE\nEnter a number: ')
	scoring_method = 'r2' if n == '1' else 'neg_mean_absolute_error'
	search = GridSearchCV(pipe, grid, scoring = scoring_method, n_jobs = -1)
	df_y = df['loan_status']

	df_x = df_x.sample(frac=0.1, random_state=42)
	df_y = df_y.loc[df_x.index]
	print('Fitting...')
	search.fit(df_x, df_y)
	print('\nr^2: ' + str(search.best_score_)) if n == '1' else print('\nMAE: ' + str(search.best_score_))

	test_name = input('\nEnter name of file containing test data: ')
	test_df = pd.read_csv(test_name)
	test_df = pd.get_dummies(test_df)
	for col in test_df.columns:
		print('filling NaN values...')
		test_df[col].fillna(test_df[col].mean(), inplace=True)
	best_pipe = search.best_estimator_
	predicted_ys = best_pipe.predict(test_df)
	for i in range(len(predicted_ys)): predicted_ys[i] = round(predicted_ys[i])
	result_df = pd.DataFrame({'id': test_df['id'], 'loan_status': predicted_ys})
	result_df.to_csv('our_submission.csv', index=False)

def gradient_boosting(df, df_x):
	steps = [('column_select', SelectColumns(df_x.columns)), ('gradient_boosting_regressor', GradientBoostingRegressor(random_state=42))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(df_x.columns[1:index]) for index in range(2, 12)],
			'gradient_boosting_regressor__n_estimators': [50, 100, 150],
			'gradient_boosting_regressor__learning_rate': [0.01, 0.1, 0.2]}
	score(df, df_x, pipe, grid)

def linear_regression(df, df_x):
	steps = [('column_select', SelectColumns(df_x.columns)), ('scaler', MinMaxScaler()), ('linear_regression', LinearRegression(n_jobs = -1))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(df_x.columns[1:index]) for index in range(2, 12)],
			'linear_regression': [LinearRegression(n_jobs = -1),
								TransformedTargetRegressor(LinearRegression(n_jobs = -1), func = np.sqrt, inverse_func = np.square),
								TransformedTargetRegressor(LinearRegression(n_jobs = -1), func = np.cbrt, inverse_func = lambda y: np.power(y, 3))]}
	score(df, df_x, pipe, grid)

def random_forest(df, df_x):
	steps = [('column_select', SelectColumns(df_x.columns)), ('scaler', MinMaxScaler()), ('random_forest_regressor', RandomForestRegressor(random_state=42))]
	pipe = Pipeline(steps)
	grid = {'column_select__columns': [list(df_x.columns[1:index]) for index in range(2, 12)],
			'random_forest_regressor__n_estimators': [50, 100],
			'random_forest_regressor__max_depth': [None, 10]}
	score(df, df_x, pipe, grid)

def main():
	handler = ErrorHandler()
	try:
		file_name = input('\nEnter name of file containing training data: ')
		with open(file_name, 'r') as file:
			df = pd.read_csv(file_name)
			df = pd.get_dummies(df)
			for col in df.columns:
				print('filling NaN values...')
				df[col].fillna(df[col].mean(), inplace=True)
			while 1:
				n = input('\nChoose a regression model:\n1. Random forest\n2. Linear regression\n3. Gradient boosting\nEnter a number: ')
				if n in ['1', '2', '3']:
					break
				else:
					print('\nPlease enter a number 1-3')
			df_x = df.drop(columns = ['loan_status'])
			return random_forest(df, df_x) if n == '1' else linear_regression(df, df_x) if n == '2' else gradient_boosting(df, df_x)
	except FileNotFoundError:
		handler.fileNotFound(False)
		main()
	except OSError:
		handler.os()
	except SyntaxError:
		handler.syntax()
	except ArithmeticError:
		handler.arithmeticError()
	except AttributeError:
		handler.attributeError()
	except EOFError:
		handler.endOfFile()
	except FloatingPointError:
		handler.floatingPoint()
	except ImportError:
		handler.importation()
	except IndentationError:
		handler.indentation()
	except IndexError:
		handler.index()
	except KeyError:
		handler.keyboard()
	except KeyboardInterrupt:
		handler.sql()
	except UnicodeDecodeError:
		handler.unicode()
	except UnicodeEncodeError:
		handler.unicode()
	except UnicodeTranslateError:
		handler.unicode()
	except UnicodeError:
		handler.unicode()
	except ValueError():
		handler.value()
	except ZeroDivisionError:
		handler.zero()
	except Exception:
		handler.bsod()

if __name__ == '__main__':
	main()
