import pandas as pd 
 
def import_data(training,validation):
	receipe = pd.read_csv(training)
	validation = pd.read_csv(validation)
	return receipe, validation


def my_classifier_function(receipe):

	predictions = []
	for i,row in receipe.iterrows():
		if row['Sugar'] <= 19.1:
			if row['Egg'] <= 12.1:
				predictions = predictions + [1] 
			else: 
				predictions = predictions + [0] 
		else:
			if row['FlourOrOats'] <= 41.32:
				predictions = predictions + [0]
			else:
				if row['FlourOrOats'] <= 42.42:
					predictions = predictions + [1]
				else:
					predictions = predictions + [0]

	Predictions = pd.DataFrame(predictions)
	Predictions.to_csv('HW_05_Rumao_sailee_MyClassifications.csv',index=False,header=False)


def main():
	recipe, validation = import_data('D:/APP STATS/720/Homework/HW_04/Recipes_For_Release_2181_v202.csv','D:/APP STATS/720/Homework/HW_04/Recipes_For_VALIDATION_2181_RELEASED_v202.csv')
	my_classifier_function(validation)



if __name__ == '__main__':
	main()