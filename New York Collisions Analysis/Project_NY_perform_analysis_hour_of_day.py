import pandas as pd

def modify_data(queens_data):

    TIME_INTERVAL = []
    time = []
    for index, row in queens_data.iterrows():

        temp = row['TIME'].strip()[0:2]
        time.append(temp)

        if(temp == '22' or temp == '23' or temp == '0:' or temp == '1:' or temp == '2:' or temp == '3:' or temp == '4:'):
            TIME_INTERVAL.append('10 PM to 5 AM')
        if(temp == '5:' or temp == '6:' or temp == '7:' or temp == '8:' or temp == '9:'):
            TIME_INTERVAL.append('5 AM to 10 AM')
        if(temp == '10' or temp == '11'):
            TIME_INTERVAL.append('10 AM to 12 PM')
        if(temp == '12' or temp == '13' or temp == '14'):
            TIME_INTERVAL.append('12 PM to 3 PM')
        if(temp == '15' or temp == '16'):
            TIME_INTERVAL.append('3 PM to 5 PM')
        if(temp == '17' or temp == '18' or temp == '19'):
            TIME_INTERVAL.append('5 PM to 8 PM')
        if(temp == '20' or temp == '21'):
            TIME_INTERVAL.append('8 PM to 10 PM')

    print(len(TIME_INTERVAL))
    print(time)
    queens_data['TIME_INTERVAL'] = TIME_INTERVAL
    return queens_data

def perform_analysis(queens_data):
    june_2017_database = []
    june_2018_database = []
    july_2017_database = []
    july_2018_database = []
    col_names = queens_data.keys()

    for index, row in queens_data.iterrows():
        month = row['DATE'][0:2]
        year = row['DATE'][6:]
        if (month == '06'):
            if (year == '17'):
                june_2017_database.append(row)
            if (year == '18'):
                june_2018_database.append(row)
        elif (month == '07'):
            if (year == '17'):
                july_2017_database.append(row)
            if (year == '18'):
                july_2018_database.append(row)
        else:
            print('Check data for mistakes')

    june_2017_database = pd.DataFrame(june_2017_database, columns=col_names)
    june_2018_database = pd.DataFrame(june_2018_database, columns=col_names)
    july_2017_database = pd.DataFrame(july_2017_database, columns=col_names)
    july_2018_database = pd.DataFrame(july_2018_database, columns=col_names)

    return [june_2017_database, june_2018_database, july_2017_database, july_2018_database]

def perform_analysis_time(queens_data):

    list_of_dic = []
    for data in queens_data:
        dic = {}
        for index, row in data.iterrows():
            if(row['TIME_INTERVAL'] not in dic):
                dic[row['TIME_INTERVAL']] = 1
            else:
                dic[row['TIME_INTERVAL']] += 1
        list_of_dic.append(dic)

    return list_of_dic

def modify_data_time(list_of_data):

    modified_data = []
    for data in list_of_data:
        temp = []
        for key in data:
            temp.append([key, data[key]])
        modified_data.append(temp)

    return modified_data

def main():

    queens_data = pd.read_csv('QUEENS_DAY.csv')
    queens_data = modify_data(queens_data)
    # queens_data.to_csv('QUEENS_TIME.csv')
    queens_data = pd.read_csv('QUEENS_TIME.csv')
    list_of_data = perform_analysis(queens_data)
    list_of_dic = perform_analysis_time(list_of_data)
    list_of_data = modify_data_time(list_of_dic)
    attributes = ['TIME_INTERVAL', 'NUMBER OF ACCIDENTS']
    june_2017_day = pd.DataFrame(list_of_data[0], columns=attributes)
    june_2017_day.to_csv('June_2017_TIME.csv')
    june_2018_day = pd.DataFrame(list_of_data[1], columns=attributes)
    june_2018_day.to_csv('June_2018_TIME.csv')
    july_2017_day = pd.DataFrame(list_of_data[2], columns=attributes)
    july_2017_day.to_csv('July_2017_TIME.csv')
    july_2018_day = pd.DataFrame(list_of_data[3], columns=attributes)
    july_2018_day.to_csv('July_2018_TIME.csv')

if __name__ == '__main__':
    main()