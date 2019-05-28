super-tiny:
	python train.py -name super_tiny -split super_tiny -eval_steps 150 -num_epochs 15

super-tiny-find:
	python train.py -name super_tiny -split super_tiny_find -load_path saved_models/train/super_tiny-04/best.pth.tar

super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -load_path saved_models/train/super_tiny-04/best.pth.tar -topk 0 -threshold -2.333

tiny:
	python train.py -name tiny -split tiny -eval_steps 500 -num_epochs 10 -batch_size 16

tiny-find:
	python train.py -name tiny -split tiny_find -load_path saved_models/train/tiny-01/best.pth.tar 

tiny-test:
	python train.py -name tiny -split tiny_test -load_path saved_models/train/tiny-01/best.pth.tar -topk 0 -threshold -2.33

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

default:
	python train.py -name default -split train -batch_size 12 -eval_steps 100000 -load_path saved_models/train/default-03/step_100008.pth.tar

