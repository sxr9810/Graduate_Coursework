
"""

Title : HW_05 : Decision Trees
Author: Sailee Rumao (sxr9810)
Date: 09/15/2018

"""


###Approach :

## 1. Importing data using pandas and understanding the data (basics)
## 2. Calculating Gini Index, generating decision tree,making predictions on this tree
      #and calculating its accuracy on training dataset.
## 3. Writing a code that write a code for the classifier program using the guidelines in the homework.
      # (emit_header,emit_classifier,emit_trailer) and exporting the csv file of predictions on validation dataset.


#Import Packages:
import matplotlib
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# %matplotlib inline



def main():

#Importing Recipes training and validation data as a dataframe using Pandas

    receipe_data = pd.read_csv("D:/APP STATS/720/Homework/HW_04/Recipes_For_Release_2181_v202.csv")
    validation_data = pd.read_csv("D:/APP STATS/720/Homework/HW_04/Recipes_For_VALIDATION_2181_RELEASED_v202.csv")

    #Changing Muffin to 1 and Cupcake to 0 in the dataset.
    receipe_data['Type'].replace(['Muffin', 'Cupcake', 'cupcake'], [1, 0, 0], inplace=True)


    #correlation plot




    # calculate the correlation matrix
    corr = receipe_data.corr()

    # plot the heatmap
    sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.title('Correlation Plot')
    plt.show()


    # Understanding data

    print(receipe_data.shape)
    print(receipe_data.columns)
    print(receipe_data.index)

    print(receipe_data.describe())

    # Histogram

    # receipe_data.hist()
    # plt.show()


    #calling the decision tree function
    emit_decision_tree(receipe_data)

    #calling the accuracy function on the training data.
    accuracy(receipe_data)
    #
    # accuracy(validation_data)


    #calling the functions to write a code for the classifier program
    header_string = emit_header()
    classifier_string = emit_classifier_Call()
    trailer_string = emit_trailer()


    #opening the file pointer for the classifier program.
    with open ('HW_05_Rumao_Sailee_Classifier.py','w') as f:
        f.write(header_string + classifier_string + trailer_string)

    #calling the file that has been written by the above code.
    import HW_05_Rumao_Sailee_Classifier
    HW_05_Rumao_Sailee_Classifier.main()


#Function to calculate weighted Gini Index.

def Gini_Index_function(receipe_data):
    Prob_0 = len(receipe_data[receipe_data['Type'] == 0]) / len(receipe_data)
    Prob_1 = len(receipe_data[receipe_data['Type'] == 1]) / len(receipe_data)
    Gini_Idx = 1 - (math.pow(Prob_1, 2) + math.pow(Prob_0, 2))
    return Gini_Idx

#Function to return class with majority datapoints.
def Return_class(receipe_data):

    if len(receipe_data[receipe_data['Type'] == 0]) > len(receipe_data[receipe_data['Type'] == 1]):
        return 0
    else:
        return 1

#Function for the decision tree algorithm.
def emit_decision_tree(receipe_data):

    #set the majority class to return
    output = Return_class(receipe_data)

    #Setting stopping criteria. i.e stop if the datapoints are less than 5 points or if data contains more than 90%
    #datapoints which belong to one the two classes.
    if len(receipe_data) < 5 or len(receipe_data[receipe_data.Type==0]) > 0.90*(len(receipe_data)) \
            or len(receipe_data[receipe_data.Type==1]) > 0.90*(len(receipe_data)):
        print('Class' , output)
        return
    else:
        #Initializing the initial combination of the best split
        Minimum_Gini_Index_at_threshold = math.inf
        best_threshold = math.inf
        best_attribute = ''
        Attribute_data = receipe_data.loc[:,'FlourOrOats':]

        #calculating teh weighted gini index for each attribute at each threshold.
        for col_name, col_values in Attribute_data.iteritems():
            #calculating threshold for each of the attribute
            Thresholds = np.arange((col_values.min())+0.1, (col_values.max())-0.1, 1)
            for threshold in Thresholds:
                Partition_1 = receipe_data[receipe_data[col_name] < threshold]
                Gini_Index_1 = Gini_Index_function(Partition_1)
                Partition_2 = receipe_data[receipe_data[col_name] > threshold]
                Gini_Index_2 = Gini_Index_function(Partition_2)
                W_Gini_Index = (len(Partition_1)/len(receipe_data))*Gini_Index_1\
                                   + (len(Partition_2)/len(receipe_data))* Gini_Index_2
                #storing the combination of best split
                if W_Gini_Index < Minimum_Gini_Index_at_threshold:
                    Minimum_Gini_Index_at_threshold = W_Gini_Index
                    best_threshold = threshold
                    best_attribute = col_name


        print(Minimum_Gini_Index_at_threshold)
        print(best_threshold)
        print(best_attribute)

    #Splitting the data based on this best split obtained above
    Data_Left = receipe_data[receipe_data[best_attribute] < best_threshold]
    Data_Right = receipe_data[receipe_data[best_attribute] >= best_threshold]
    # calling the function recursively on the left data and printing instructions
    print('Left split')
    emit_decision_tree(Data_Left)
    # calling the function recursively on the right data and printing instructions
    print('Right split')
    emit_decision_tree(Data_Right)

#Function to calculate predictions and the accuracy based on the decision trees generated above
def accuracy(receipe_data):

    predictions = []
    for i,row in receipe_data.iterrows():
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

    print(predictions)
    print(len(predictions))

    #Extracting the Target variable Type from the receipe data as a list.
    Actual = receipe_data['Type'].tolist()

    #calculating accuracy
    True_positive = 0
    True_Negative = 0
    for ind in range(len(Actual)):
        if Actual[ind] == predictions[ind]:
            if predictions[ind] == 1:
                True_positive +=1
            else:
                True_Negative +=1
    print(True_positive)
    print(True_Negative)
    Accuracy = (True_positive+True_Negative)/len(receipe_data)
    print('The Accuracy is:',Accuracy)

#Function to emit code for the header part of the classifier program.
def emit_header():

    header_string = ''
    header_string += 'import pandas as pd \n \n' \
                     'def import_data(training,validation):\n\t' \
                     'receipe = pd.read_csv(training)\n\t' \
                     'validation = pd.read_csv(validation)\n\t' \
                     'return receipe, validation\n\n\n'

    return header_string

#Function to emit code for the classifier part of the classifier program.
#From this code, this function will make the required predictions on the validation dataset
#and export them as csv file in the classifier program.
def emit_classifier_Call():

    classifier_string = ""
    classifier_string += "def my_classifier_function(receipe):\n\n\t" \
                         "predictions = []\n\t" \
                         "for i,row in receipe.iterrows():\n\t\t" \
                         "if row['Sugar'] <= 19.1:\n\t\t\t" \
                         "if row['Egg'] <= 12.1:\n\t\t\t\t" \
                         "predictions = predictions + [1] \n\t\t\t" \
                         "else: \n\t\t\t\t" \
                         "predictions = predictions + [0] \n\t\t" \
                         "else:\n\t\t\t" \
                         "if row['FlourOrOats'] <= 41.32:\n\t\t\t\t" \
                         "predictions = predictions + [0]\n\t\t\t" \
                         "else:\n\t\t\t\t" \
                         "if row['FlourOrOats'] <= 42.42:\n\t\t\t\t\t" \
                         "predictions = predictions + [1]\n\t\t\t\t" \
                         "else:\n\t\t\t\t\t" \
                         "predictions = predictions + [0]\n\n\t" \
                         "Predictions = pd.DataFrame(predictions)\n\t" \
                         "Predictions.to_csv('HW_05_Rumao_sailee_MyClassifications.csv',index=False,header=False)\n\n\n"

    return classifier_string


#Function to emit code for the trailer part that is the main function etc. in the classifier program.
def emit_trailer():

    trailer_string = ""
    trailer_string += "def main():\n\t" \
                      "recipe, validation = import_data('D:/APP STATS/720/Homework/HW_04/Recipes_For_Release_2181_v202.csv'," \
                      "'D:/APP STATS/720/Homework/HW_04/Recipes_For_VALIDATION_2181_RELEASED_v202.csv')\n\t" \
                      "my_classifier_function(validation)\n\n\n\n"\
                      "if __name__ == '__main__':\n\t" \
                      "main()"

    return trailer_string



if __name__ == '__main__':
  main()
