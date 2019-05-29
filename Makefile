super-tiny:
	python train.py -name super_tiny -split super_tiny -gpu_ids cpu -eval_steps 150 -num_epochs 25

super-tiny-test-dev:
	python train.py -name super_tiny -split super_tiny_test_dev -gpu_ids cpu -load_path saved_models/train/super_tiny-02/best.pth.tar

super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-02/best.pth.tar

tiny:
	python train.py -name tiny -split tiny -eval_steps 500 -num_epochs 10

tiny-test-dev:
	python train.py -name tiny -split tiny_test_dev -load_path # fill last param

test:
	python train.py -name default -split test_test -load_path # fill last param

dirs:
	mkdir data
	mkdir saved_models

clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf saved_models/*
train:
	python train.py -name default -split train -batch_size 12 -eval_steps 100000
default: train

