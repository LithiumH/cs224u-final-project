super-tiny:
	python train.py -name super_tiny -split super_tiny -gpu_ids cpu -eval_steps 500 -num_epochs 50
super-tiny-test:
	python train.py -name super_tiny -split super_tiny_test -gpu_ids cpu -load_path saved_models/train/super_tiny-13/best.pth.tar
tiny:
	python train.py -name tiny -split tiny
test:
	python train.py -name default -split test
default:
	python train.py -name default -split train
