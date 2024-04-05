## TEXT TO SQL
create virtual env
```bash 
$ python -m venv venv

$ source venv/bin/activate

$ pip install -r requirements.txt
```

1. Create Data for pretraining
```bash
$ python src-llm/run.py -c data_prep
```

2. Pretraining a foundation model

```bash
$ python src-llm/run.py -c pretrain_and_save
```
3. if you want to just run the code and donot wish to save the model run the following cmd

```bash
$ python src-llm/run.py -c pretrain
```

To run chainlit db integration use server.py in src folder 

