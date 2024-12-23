import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class SelectColumns(BaseEstimator, TransformerMixin):
	def __init__(self, columns):
		self.columns = columns

	def fit(self, xs, ys, **params):
		return self

	def transform(self, xs):
		return xs[self.columns]

def score_classifier(df, df_x, pipe, grid):
	n = input('\nChoose scoring:\n1. accuracy\n2. precision\nEnter a number: ')
	scoring_method = 'accuracy' if n == '1' else 'average_precision'
	search = GridSearchCV(pipe, grid, scoring = scoring_method, n_jobs = -1)
	df_y = df['loan_status']

	df_x = df_x.sample(frac=0.1, random_state=42)
	df_y = df_y.loc[df_x.index]
	print('Fitting...')
	search.fit(df_x, df_y)
	print('\naccuracy: ' + str(search.best_score_)) if n == '1' else print('\nprecission: ' + str(search.best_score_))

	try:
		test_name = input('\nEnter name of file containing test data: ')
		test_df = pd.read_csv(test_name)
		test_df = pd.get_dummies(test_df)
		print('filling NaN values...')
		for col in test_df.columns:
			test_df[col].fillna(test_df[col].mean())
		best_pipe = search.best_estimator_
		predicted_ys = best_pipe.predict(test_df)
		for i in range(len(predicted_ys)): predicted_ys[i] = round(predicted_ys[i])
		result_df = pd.DataFrame({'id': test_df['id'], 'loan_status': predicted_ys})
		result_df.to_csv('our_submission.csv', index=False)
		print('Operation successful, results written to: ' + test_name)

	except FileNotFoundError:
		print("Couldn't find "+test_name)
		quit()

def score_regressor(df, df_x, pipe, grid):
	n = input('\nChoose scoring:\n1. r^2\n2. MAE\nEnter a number: ')
	scoring_method = 'r2' if n == '1' else 'neg_mean_absolute_error'
	search = GridSearchCV(pipe, grid, scoring = scoring_method, n_jobs = -1)
	df_y = df['loan_status']

	df_x = df_x.sample(frac=0.1, random_state=42)
	df_y = df_y.loc[df_x.index]
	print('Fitting...')
	search.fit(df_x, df_y)
	print('\nr^2: ' + str(search.best_score_)) if n == '1' else print('\nMAE: ' + str(search.best_score_))

	try:
		test_name = input('\nEnter name of file containing test data: ')
		test_df = pd.read_csv(test_name)
		test_df = pd.get_dummies(test_df)
		print('filling NaN values...')
		for col in test_df.columns:
			test_df[col].fillna(test_df[col].mean())
		best_pipe = search.best_estimator_
		predicted_ys = best_pipe.predict(test_df)
		for i in range(len(predicted_ys)): predicted_ys[i] = round(predicted_ys[i])
		result_df = pd.DataFrame({'id': test_df['id'], 'loan_status': predicted_ys})
		result_df.to_csv('our_submission.csv', index=False)
		print('Operation successful, results written to: ' + test_name)

	except FileNotFoundError:
		print("Couldn't find "+test_name)
		quit()

def gradient_boosting(df, df_x):
	pipe = Pipeline([
		('column_select', SelectColumns(None)),
		('gradient_boosting_classifier', GradientBoostingClassifier(random_state=42))
	])
	grid = {
		'column_select__columns': 
			[list(df_x.columns[1:index]) for index in range(2, 12)],
		'gradient_boosting_classifier__n_estimators': 
			[50, 100, 150],
		'gradient_boosting_classifier__learning_rate': 
			[0.01, 0.1, 0.2]
	}
	score_classifier(df, df_x, pipe, grid)

def random_forest(df, df_x):
	pipe = Pipeline([
		('column_select', SelectColumns(df_x.columns)), 
		('scaler', MinMaxScaler()), ('random_forest_classifier', RandomForestClassifier(random_state=42))
	])
	grid = {
		'column_select__columns': 
			[list(df_x.columns[1:index]) for index in range(2, 12)],
		'random_forest_classifier__n_estimators': 
			[50, 100],
		'random_forest_classifier__max_depth': 
			[None, 10]
	}
	score_classifier(df, df_x, pipe, grid)

def linear_regression(df, df_x):
	pipe = Pipeline([
		('column_select', SelectColumns(df_x.columns)), 
		('scaler', MinMaxScaler()), ('linear_regression', LinearRegression(n_jobs = -1))
	])
	grid = {
		'column_select__columns': 
			[list(df_x.columns[1:index]) for index in range(2, 12)],
		'linear_regression': 
			[LinearRegression(n_jobs = -1), 
				TransformedTargetRegressor(LinearRegression(n_jobs = -1), 
					func = np.sqrt, inverse_func = np.square),
				TransformedTargetRegressor(LinearRegression(n_jobs = -1), 
					func = np.cbrt, inverse_func = lambda y: np.power(y, 3))
			]
	}
	score_regressor(df, df_x, pipe, grid)

def main():
	file_name = input('\nEnter name of file containing training data: ')
	try:
		with open(file_name, 'r') as file:
			df = pd.read_csv(file_name)
			df = pd.get_dummies(df)
			print('filling NaN values...')
			for col in df.columns:
				df[col].fillna(df[col].mean())
			while 1:
				n = input('\nChoose a model:\n1. Random forest classifier\n2. Linear regression\n3. Gradient boosting classifier\nEnter a number: ')
				if n in ['1', '2', '3']:
					break
				else:
					print('\nPlease enter a number 1-3')
			df_x = df.drop(columns = ['loan_status'])
			return random_forest(df, df_x) if n == '1' else linear_regression(df, df_x) if n == '2' else gradient_boosting(df, df_x)
	
	except FileNotFoundError:
		print('Could not find '+file_name)
		main()

if __name__ == '__main__':
	main()