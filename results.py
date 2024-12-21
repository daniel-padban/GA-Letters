import plotly.graph_objects as go
import wandb
import pandas as pd
import numpy as np
import os
import plotly.io as pio
 
def download_wandb_project_data(entity_name:str, project_name:str,group:str):
    """
    Download data from a W&B project.
    
    Args:
        entity_name (str): The W&B entity (username or team name)
        project_name (str): The name of the W&B project
        
    Returns:
        dict: Dictionary containing project data with runs, histories, configs, and summaries
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Construct project path
    project_path = f"{entity_name}/{project_name}"
    
    try:
        # Access the project
        project = api.project(project_path)
        
        # Get all runs in the project
        runs = api.runs(project_path, filters={"group": group})
        
        # Initialize dictionary to store all data
        project_data = {
            'runs': [],
            'histories': [],
            'configs': [],
            'summaries': []
        }
        
        # Download data from each run
        run_names = []
        for run in runs:
            print(f"Downloading data from run: {run.name}")
            
            # Get run history
            history = run.history()
            rel_history = history[['_step','train_step','train_F1','train_accuracy','train_CO_loss','test_step','test_F1','test_accuracy','test_CO_loss']]
            # Get run config
            config = run.config
            
            # Get run summary
            summary = run.summary
            
            # Store data in dictionary
            project_data['runs'].append(run)
            project_data['histories'].append(rel_history)
            project_data['configs'].append(config)
            project_data['summaries'].append(summary)
            run_names.append(run.name)
        
        return project_data, run_names
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def transform_data(joined_data_list:list[pd.DataFrame]=None,tuple_data_list:list[tuple[pd.DataFrame,pd.DataFrame]]=None,names:list[str]=None,group:str=None,to_csv:bool=False):
    '''
    Returns data list with tuples of shape (test_data,train_data)
    '''
    transformed_data_list = []
    
    cols_to_drop = ['_step','test_step','train_step']
    if joined_data_list is not None:
        for i,data in enumerate(joined_data_list):
            data = data.drop(columns=cols_to_drop)

            test_data = data[['test_F1','test_accuracy','test_CO_loss']]
            train_data = data[['train_F1','train_accuracy','train_CO_loss']]
            test_data = test_data.dropna().reindex()
            train_data = train_data.dropna().reindex()

            if names is not None and to_csv and group is not None:
                test_path = f'results/{group}/test_{names[i]}.csv'
                train_path = f'results/{group}/train_{names[i]}.csv'
                test_data.to_csv(test_path)
                train_data.to_csv(train_path)
            transformed_data_list.append((test_data,train_data))
    else:
        transformed_data_list = [(tuple_data_list[i][0].drop(columns='Unnamed: 0'),tuple_data_list[i][1].drop(columns='Unnamed: 0')) for i in range(len(tuple_data_list))]

    return transformed_data_list

def aggregate_data(data_list:list[pd.DataFrame]):
    numpy_data_list = [df.to_numpy() for df in data_list]
    stacked_data = np.stack(numpy_data_list,0)
    agg_data:np.ndarray = stacked_data.mean(0)
    # Generate an index column starting from 1
    index_column = np.arange(1, agg_data.shape[0] + 1).reshape(-1, 1)

    # Add the index column to the ndarray
    agg_data = np.hstack((index_column, agg_data))

    return agg_data

def create_data2plot(data_list:list[np.ndarray],sizes):

    '''
    Returns:
    
    (x,y,z)
    
    (sizes, steps, metric)
    '''
    x_vals = [] #dataset size
    y_vals = [] #step
    z_vals = [] #performance
    for i, array in enumerate(data_list):
        x_vals.append(np.full_like(array[:, 1], sizes[i]))  # Set x values (from List 1), dataset size
        y_vals.append(array[:, 0])  # Extract y values (2nd column), step
        z_vals.append(array[:, 1])  # Extract z values (1st column), metric

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    z_arr = np.array(z_vals)

    ''' x_mesh, y_mesh = np.meshgrid(x_arr.flatten(), np.meshgrid(y_arr.flatten()))
    z_mesh = np.meshgrid(z_arr.flatten(),z_arr.flatten())'''

    return x_arr, y_arr, z_arr

def create_3d_plot(x, y, z, xlabel, ylabel, zlabel, title, filename):
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='emrld')])
        fig.update_layout(
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                zaxis_title=zlabel,
            ),
            title=title
        )
        pio.write_html(fig, file=filename)
        return fig

def last_mean(data:np.ndarray,last_n):
    last_data:np.ndarray = data[:,-last_n:]
    last_mean = last_data.mean(axis=1)
    return last_mean

def create_line_plot(x:np.ndarray, test_y, train_y, xlabel, ylabel, title, filename):
        x = x[:,0]
        test_y = test_y
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x,y=test_y, mode='lines',name=f'Test {ylabel}')
        )
        fig.add_trace(
            go.Scatter(x=x,y=train_y, mode='lines',name=f'Train {ylabel}')
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
            ),
            title=title
        )
        pio.write_html(fig, file=filename)
        return fig

if __name__ == "__main__":
    groups = [f'B{i}' for i in range(1,12)]
    test_aggs = {}
    train_aggs = {}
    train_sizes = []
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

    for group in groups:
        print(group)
        '''data,names = download_wandb_project_data("DP-Team", "GA-Letters", group=group)
        histories = data['histories']'''

        names = [f'Run-D{group_sizes[group]}-S{100*i+i}' for i in range(10)]
        
        file_groups = {}
        for filename in os.listdir(f'results/{group}'):
            # Extract the prefix and base name
            if filename.startswith("test_"):
                base_name = filename[5:]  # Remove "test_" prefix
                file_groups.setdefault(base_name, {})["test"] = filename
            elif filename.startswith("train_"):
                base_name = filename[6:]  # Remove "train_" prefix
                file_groups.setdefault(base_name, {})["train"] = filename

# Create tuples of (test_df, train_df)
        histories = []
        for base_name, files in file_groups.items():
            if "test" in files and "train" in files:
                test_file_path = os.path.join('results', group, files["test"])
                train_file_path = os.path.join('results', group, files["train"])
                
                # Read dataframes
                test_df = pd.read_csv(test_file_path)
                train_df = pd.read_csv(train_file_path)
                
                # Append tuple to histories
                histories.append((test_df, train_df))
        
        train_size = group_sizes[group]
        train_sizes.append(train_size)
        #test:pd.DataFrame = pd.read_csv('results/test.csv')
        
        transformed_data_list = transform_data(tuple_data_list=histories,group=group,names=names,to_csv=True)
        test_transformed = [data_tuple[0] for data_tuple in transformed_data_list]
        train_transformed = [data_tuple[1] for data_tuple in transformed_data_list]
        
        test_column_index = {index:col for index,col in enumerate(test_transformed[0].columns)} #debug purpose
        train_column_index = {index:col for index,col in enumerate(train_transformed[0].columns)} #debug purpose

        test_agg = aggregate_data(test_transformed)
        train_agg = aggregate_data(train_transformed)

        test_aggs[group] = test_agg
        train_aggs[group] = train_agg

    test_F1s = [test_aggs[group][:,[0,1]] for group in groups]
    test_accs = [test_aggs[group][:,[0,2]] for group in groups] # *100 for percentage
    test_CE_losses =[test_aggs[group][:,[0,3]] for group in groups]

    train_F1s = [train_aggs[group][:,[0,1]] for group in groups]
    train_accs = [train_aggs[group][:,[0,2]] for group in groups] # *100 for percentage
    train_CE_losses =[train_aggs[group][:,[0,3]] for group in groups]

    test_f1_x,test_f1_y,test_f1_z = create_data2plot(test_F1s,train_sizes)
    test_acc_x, test_acc_y, test_acc_z = create_data2plot(test_accs,train_sizes)
    test_CE_x, test_CE_y, test_CE_z = create_data2plot(test_CE_losses,train_sizes)

    train_f1_x,train_f1_y,train_f1_z = create_data2plot(train_F1s,train_sizes)
    train_acc_x, train_acc_y, train_acc_z = create_data2plot(train_accs,train_sizes)
    train_CE_x, train_CE_y, train_CE_z = create_data2plot(train_CE_losses,train_sizes)

# Generate and save the plots
    plotly_test_F1 = create_3d_plot(test_f1_x, test_f1_y, test_f1_z, 
                                    'Size', 'Steps', 'F1', 
                                    'Test F1, Dataset Size, Steps', 
                                    "graphs/test_f1_plot.html")

    plotly_test_acc = create_3d_plot(test_acc_x, test_acc_y, test_acc_z*100, 
                                    'Size', 'Steps', 'Accuracy %', 
                                    'Test Accuracy, Dataset Size, Steps', 
                                    "graphs/test_acc_plot.html")

    plotly_test_CE = create_3d_plot(test_CE_x, test_CE_y, test_CE_z, 
                                    'Size', 'Steps', 'Cross Entropy Loss', 
                                    'Test Cross Entropy, Dataset Size, Steps', 
                                    "graphs/test_CE_plot.html")

    plotly_train_F1 = create_3d_plot(train_f1_x, train_f1_y, train_f1_z, 
                                    'Size', 'Steps', 'F1', 
                                    'Train F1, Dataset Size, Steps', 
                                    "graphs/train_f1_plot.html")

    plotly_train_acc = create_3d_plot(train_acc_x, train_acc_y, train_acc_z*100, 
                                    'Size', 'Steps', 'Accuracy %', 
                                    'Train Accuracy, Dataset Size, Steps', 
                                    "graphs/train_acc_plot.html")

    plotly_train_CE = create_3d_plot(train_CE_x, train_CE_y, train_CE_z, 
                                    'Size', 'Steps', 'Cross Entropy Loss', 
                                    'Train Cross Entropy, Dataset Size, Steps', 
                                    "graphs/train_CE_plot.html")

    plotly_2d_F1 = create_line_plot(x=test_CE_x,test_y=last_mean(test_f1_z,3),train_y=last_mean(train_f1_z,3),xlabel='Size',ylabel='F1',title='F1',filename='graphs/2d_F1_plot.html')
    plotly_2d_acc = create_line_plot(x=test_CE_x,test_y=last_mean(test_acc_z*100,3),train_y=last_mean(train_acc_z*100,3),xlabel='Size',ylabel='Accuracy %',title='F1',filename='graphs/2d_acc_plot.html')
    plotly_2d_CE = create_line_plot(x=test_CE_x,test_y=last_mean(test_CE_z,3),train_y=last_mean(train_CE_z,3),xlabel='Size',ylabel='F1',title='Cross Entropy',filename='graphs/2d_CE_plot.html')
    
    plotly_2d_F1.show()
    plotly_test_F1.show()


    # Display one of the plots (optional)

