#run tests for using all labels for varied batchsize
python -m src.train_baseline wine CPU:0 4000 64;
python -m src.train_baseline wine CPU:0 4000 128;
python -m src.train_baseline wine CPU:0 4000 256;
#run tests for using 2000 labels for varied batchsize
python -m src.train_baseline wine CPU:0 2000 64;
python -m src.train_baseline wine CPU:0 2000 128;
python -m src.train_baseline wine CPU:0 2000 256;
#run tests for using 1000 labels for varied batchsize
python -m src.train_baseline wine CPU:0 1000 64;
python -m src.train_baseline wine CPU:0 1000 128;
python -m src.train_baseline wine CPU:0 1000 256;
#run tests for using 500 labels for varied batchsize
python -m src.train_baseline wine CPU:0 500 64;
python -m src.train_baseline wine CPU:0 500 128;
python -m src.train_baseline wine CPU:0 500 256;
#run tests for using 250 labels for varied batchsize
python -m src.train_baseline wine CPU:0 250 64;
python -m src.train_baseline wine CPU:0 250 128;
python -m src.train_baseline wine CPU:0 250 256;