import wandb
 
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
        for run in runs:
            print(f"Downloading data from run: {run.name}")
            
            # Get run history
            history = run.history()
            
            
            # Get run config
            config = run.config
            
            # Get run summary
            summary = run.summary
            
            # Store data in dictionary
            project_data['runs'].append(run)
            project_data['histories'].append(history)
            project_data['configs'].append(config)
            project_data['summaries'].append(summary)
        
        return project_data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    data = download_wandb_project_data("DP-Team", "GA-Letters", group="D5")
    print(data)
