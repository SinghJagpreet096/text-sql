import click
import pretrainer_multipcore
import prepare_data_openweb
import time
from components.utils import processing_time

DATA_PREP = "data_prep"
PRETRAIN = "pretrain"
PRETRAIN_SAVE = "pretrain_and_save"
FINE_TUNE = "fine-tune"


@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice([DATA_PREP,PRETRAIN,PRETRAIN_SAVE, FINE_TUNE]), 
    help="Choose the pipe to run"
    "data_prep: Prepare data"
    "pretrain: Pretrain the model"
    "fine-tune: Fine-tune the model")

def run_pipelines(config:str):
    start_time = time.time()
    if config == DATA_PREP:
        prepare_data_openweb.main()
    elif config == PRETRAIN:
        pretrainer_multipcore.main()
    elif config == PRETRAIN_SAVE:
        pretrainer_multipcore.main(save=True)
    elif config == FINE_TUNE:
        pass
    else:
        raise ValueError(f"Invalid config: {config}")
    end_time = time.time()
    print(f"Pipeline {config} completed")
    print(f'{processing_time(start_time,end_time)}')  
    exit(0) 

if __name__ == "__main__":
    run_pipelines() 
   
    