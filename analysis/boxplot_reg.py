import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm





def create_and_save_boxplot(df:pd.DataFrame, x_column, y_column, output_file,ylabel:str,log:bool=False,train:bool=False):
    """
    Creates a boxplot grouped by a column and saves it as an HTML file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - x_column (str): Column name for the x-axis (grouping).
    - y_column (str): Column name for the y-axis (values to plot).
    - output_file (str): File path to save the HTML plot.
    """
    if train:
        train_state = 'train'
    else:
        train_state = 'test'
    fig = px.box(df, 
                x=x_column, 
                y=y_column, 
                log_x=True,
                title=f'Box plot of {train_state} {ylabel.lower()} and dataset size (logarithmic)',
                labels={
                    "log_dataset": "Log base 10 of dataset size",
                    y_column: ylabel,
                },
                )

    fig.write_html(output_file)
    print(f"Boxplot saved to {output_file}")



dataset_dfs = []
for i in range(1,12):
    dataset_path = os.path.join('results',f'B{i}')
    runs = os.listdir(dataset_path)
    train_paths = [run for run in runs if re.match('train.*',run)]
    test_paths = [run for run in runs if re.match('test.*',run)]
    
    joined_dfs = []
    for j in range(10):
        train_df = pd.read_csv(os.path.join(dataset_path,train_paths[j]))
        test_df = pd.read_csv(os.path.join(dataset_path,test_paths[j]))
        train_last=train_df.iloc[-1,-3:].reset_index()
        test_last=test_df.iloc[-1,-3:].reset_index()

        joined_df = pd.concat([train_last,test_last],axis=0,ignore_index=True).set_index('index').T
        joined_dfs.append(joined_df)
        
    dataset_df = pd.concat(joined_dfs,axis=0).reset_index(drop=True)
    dataset_df['dataset'] = f'B{i}'
    dataset_dfs.append(dataset_df)

all_data = pd.concat(dataset_dfs)
group_sizes = {
        'B1':12000,
        'B2':6000,
        'B3':4000,
        'B4':3000,
        'B5':2400,
        'B6':2000,
        'B7':1500,
        'B8':1200,
        'B9':1000,
        'B10':800,
        'B11':750,
    }
group_sizes = {key:value for key, value in group_sizes.items()}
all_data.replace(group_sizes,inplace=True)
all_data['log_dataset'] = np.log10(all_data['dataset'])

#Graphs
train_f1_plot = create_and_save_boxplot(all_data[['train_F1','log_dataset']],'log_dataset','train_F1','graphs/box_train_f1.html',ylabel='F1',log=True,train=True)
train_acc_plot = create_and_save_boxplot(all_data[['train_accuracy','log_dataset']],'log_dataset','train_accuracy','graphs/box_train_accuracy.html',ylabel='Accuracy %',log=True,train=True)
train_CE_plot = create_and_save_boxplot(all_data[['train_CO_loss','log_dataset']],'log_dataset','train_CO_loss','graphs/box_train_CE.html',ylabel='Cross Entropy',log=True,train=True)

train_f1_plot = create_and_save_boxplot(all_data[['test_F1','log_dataset']],'log_dataset','test_F1','graphs/box_test_F1.html',ylabel='F1',log=True)
train_f1_plot = create_and_save_boxplot(all_data[['test_accuracy','log_dataset']],'log_dataset','test_accuracy','graphs/box_test_accuracy.html',ylabel='Accuracy %',log=True)
train_f1_plot = create_and_save_boxplot(all_data[['test_CO_loss','log_dataset']],'log_dataset','test_CO_loss','graphs/box_test_CE.html',ylabel='Cross Entropy',log=True)

#Regression
aggregated_data = all_data.groupby('dataset').mean().reset_index()
aggregated_data['log_dataset'] = np.log10(aggregated_data['dataset'])

aggregated_data.rename(columns={'train_CO_loss':'train_cross_entropy','test_CO_loss':'test_cross_entropy'},inplace=True)

X = sm.add_constant(aggregated_data['log_dataset'])  

train_F1 = aggregated_data['train_F1']
train_acc = aggregated_data['train_accuracy']
train_CE = aggregated_data['train_cross_entropy']

test_F1 = aggregated_data['test_F1']
test_acc = aggregated_data['test_accuracy']
test_CE = aggregated_data['test_cross_entropy']

# Fit linear regression model
train_F1_model = sm.OLS(train_F1, X).fit()
train_acc_model = sm.OLS(train_acc, X).fit()
train_CE_model = sm.OLS(train_CE, X).fit()

test_F1_model = sm.OLS(test_F1, X).fit()
test_acc_model = sm.OLS(test_acc, X).fit()
test_CE_model = sm.OLS(test_CE, X).fit()
reg_models = [train_F1_model,train_acc_model,train_CE_model,test_F1_model,test_acc_model,test_CE_model]
reg_summaries = [model.summary().as_text() for model in reg_models]

#Write results to file
with open('regressions.txt','w') as reg_file:
    for i in range(len(reg_summaries)):
        reg_file.write('\n'+reg_summaries[i]+'\n')



