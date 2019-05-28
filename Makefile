super-tiny:
	python train.py -name super_tiny -split super_tiny -gpu_ids cpu -eval_steps 150 -num_epochs 15
super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-13/best.pth.tar
tiny:
	python train.py -name tiny -split tiny -batch_size 12 -eval_steps 500 -num_epochs 10
test:
	python train.py -name default -split test
dirs:
	mkdir data
	mkdir saved_models
clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf saved_models/*
train:
	python train.py -name default -split train -batch_size 12 -load_path saved_models/train/default-01/step_100008.pth.tar -eval_steps 100000
default: train

