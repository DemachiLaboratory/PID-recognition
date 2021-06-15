#python train.py --batch-size 16 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-csp.yaml --weights 'pretrain/yolov4-csp.weights' --sync-bn --device 0 --epochs 400  --name yolov4-csp --logdir /mnt/data/Experiments/20210504-pid-csp-640/
#python train.py --batch-size 16 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --device 0 --epochs 400  --name yolov4-p7 --logdir /mnt/data/Experiments/20210504-pid-p7-640/
#python train.py --batch-size 8 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --multi-scale --device 0 --epochs 400  --name yolov4-p7 --logdir /mnt/data/Experiments/20210504-pid-p7-640-ms/
#python train.py --batch-size 16 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p5.yaml --weights 'pretrain/yolov4-p5.pt' --sync-bn --device 0 --epochs 400  --name yolov4-p5 --logdir /mnt/data/Experiments/20210504-pid-p5-640/
#python train.py --batch-size 16 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p6.yaml --weights 'pretrain/yolov4-p6.pt' --sync-bn --device 0 --epochs 400  --name yolov4-p6 --logdir /mnt/data/Experiments/20210504-pid-p6-640/
python train.py --multi-scale --batch-size 8 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p5.yaml --weights 'pretrain/yolov4-p5.pt' --sync-bn --device 0 --epochs 400  --name yolov4-p5 --logdir /mnt/data/Experiments/20210505-pid-p5-640_ms/
python train.py --multi-scale --batch-size 4 --img 640 640 --data pid.yaml --cfg models/pid/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --device 0 --epochs 400  --name yolov4-p7 --logdir /mnt/data/Experiments/20210505-pid-p7-640_ms/