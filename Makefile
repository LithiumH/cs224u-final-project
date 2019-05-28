super-tiny:
	python train.py -name super_tiny -split super_tiny -gpu_ids cpu -eval_steps 150 -num_epochs 15

super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-02/best.pth.tar

super-tiny-find:
	python train.py -name super_tiny -split super_tiny_find -gpu_ids cpu -load_path saved_models/train/super_tiny-02/best.pth.tar

super-tiny-test-thresh:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-02/best.pth.tar -topk 0 -threshold -3.0

tiny:
	python train.py -name tiny -split tiny -eval_steps 500 -num_epochs 10

tiny-find:
	python train.py -name tiny -split tiny -load_path # fill last param

find:
	python train.py -name default -split find -load_path # fill last param

test-thresh:
	python train.py -name default -split test -topk 0 -threshold -load_path # fill last 2 param

dirs:
	mkdir data
	mkdir saved_models

clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf saved_models/*
train:
	python train.py -name default -split train -batch_size 12 -eval_steps 100000
default: train

