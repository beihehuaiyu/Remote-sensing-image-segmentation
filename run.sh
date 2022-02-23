git clone https://gitee.com/paddlepaddle/PaddleSeg.git
pip install filelock
unzip /root/paddlejob/workspace/train_data/datasets/data75675/UDD5.zip -d /root/paddlejob/workspace/train_data/datasets/UDD
mv use.yml PaddleSeg/configs
cd PaddleSeg
python -m paddle.distributed.launch train.py --config configs/use.yml --do_eval \
       --use_vdl --save_interval 200 --save_dir output
