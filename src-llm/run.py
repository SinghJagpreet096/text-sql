import click
import pretrainer_multipcore
import prepare_data_openweb
import time
from components.utils import processing_time

DATA_PREP = "data-prep"
PRETRAIN = "pretrain"
FINE_TUNE = "fine-tune"


@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice([DATA_PREP,PRETRAIN, FINE_TUNE]), 
    help="Choose the pipe to run"
    "data-prep: Prepare data"
    "pretrain: Pretrain the model"
    "fine-tune: Fine-tune the model")

def run_pipelines(config:str):
    start_time = time.time()
    if config == DATA_PREP:
        prepare_data_openweb.main()
    elif config == PRETRAIN:
        pretrainer_multipcore.main()
    elif config == FINE_TUNE:
        pass
    else:
        raise ValueError(f"Invalid config: {config}")
    end_time = time.time()
    print(f"Pipeline {config} completed")
    print(f'{processing_time(start_time,end_time)}')   

if __name__ == "__main__":
    start_time = time.time()
    run_pipelines() 
    end_time = time.time()
    processing_time(start_time,end_time)    
    