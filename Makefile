super-tiny:
	python train.py -name super_tiny -split super_tiny -gpu_ids cpu -eval_steps 150 -num_epochs 15
super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-13/best.pth.tar
tiny:
	python train.py -name tiny -split tiny -eval_steps 500 -num_epochs 10
test:
	python train.py -name default -split test
dirs:
	mkdir data
	mkdir saved_models
clean:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf saved_models/*
default:
	python train.py -name default -split train
