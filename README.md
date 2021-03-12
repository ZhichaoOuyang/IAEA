# IAEA
IAEA (Integrity-Aware Extractive-Abstractive realtime event summarization) is an unified extractive-abstractive framework for realtime event summarization. Our key idea is to integrate an inconsistency detection module to preserve the integrity of the summaries in each time slice.

## HID_Model
### Requirements
* Hardwares: a machine with two Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz, 256 GB main memory and a GeForce RTX 2080 Ti graphics card
* OS: Ubuntu 18.04
* Packages:
    * python3.6 
    * tensorflow 1.13.1-gpu
    * keras 2.4.2

- Train
 ```shell script
 python HID_train.py
 ```
- Get HID pre-trained parameters
 ```shell script
 python getInconsistentWeight.py
 ```

 After the above steps, data/inconsistent_weight.npy and data/inconsistent_bias.npy are obtained for use in IAEA-Model.

 **Note**: You can directly use these .npy file we provide in `data/` folder to train IAEA-Model.


## IAEA_Model
### Requirements
* Hardwares: a machine with two Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz, 256 GB main memory and a GeForce RTX 2080 Ti graphics card
* OS: Ubuntu 18.04
* Packages:
    * python3.6 
    * tensorflow 1.2.1-gpu

**Note**: you can use the command to start a tf1.2.1-gpu docker
```shell script
docker run -itd --gpus all --name tf1.2 -v /home/dm:/workspace tensorflow/tensorflow:1.2.1-gpu-py3

docker exec -it tf1.2 /bin/bash
```
### Data preprocess

 ```shell script
python data/IAEA/make_datafiles_testdata.py data/IAEA/twitter_final/train
 ```
run this command to get train dataset, to same to valid and test dataset.


**Note**:you can skip this step and then use the processed dataset of `data/IAEA/finished_files_twitter` folder


### Extractor
 - Train
 ```shell script
python main.py --model=selector --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_gpu_selector_lr001 --exp_name=exp_sample --max_art_len=110 --max_sent_len=50 --max_train_iter=1500 --batch_size=5 --save_model_every=500 --lr=0.01 --model_max_to_keep=25
```

### Abstractor
 - Train
 ```shell script
python main.py --model=rewriter --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_rewriter --exp_name=exp_sample --max_enc_steps=400 --max_dec_steps=100 --batch_size=5 --max_train_iter=5000 --save_model_every=1000 --model_max_to_keep=5
 ```

 - Add reinforcement learning
 ```shell script
 python main.py --model=rewriter --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_rewriter --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True  --convert_to_reinforce_model=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10
 ```

 ```shell script
 python main.py --model=rewriter --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_rewriter --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10
 ```
 - Add coverage mechanism
 ```shell script
 python main.py  --model=rewriter --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_rewriter --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10 --coverage=True --convert_to_coverage_model=True
 ```

 ```shell script
 python main.py  --model=rewriter --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_rewriter --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10 --coverage=True
 ```


### End2End
- Train
 ```shell script
 python main.py --model=end2end --mode=train --data_path=data/IAEA/finished_files_twitter/chunked/train_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_gpu_endlr0001 --exp_name=exp_sample --max_enc_steps=800 --max_dec_steps=120 --max_train_iter=10000 --batch_size=5 --use_temporal_attention=True --intradecoder=True --eta=2.5E-05 --max_art_len=110 --max_sent_len=50 --selector_loss_wt=5.0 --inconsistent_loss=True --inconsistent_topk=3 --save_model_every=1000 --model_max_to_keep=20 --rl_training=True --coverage=True --pretrained_selector_path=log_gpu_selector_lr001/selector/exp_sample/train/model.ckpt-500 --pretrained_rewriter_path=log_rewriter/rewriter/exp_sample/train/model.ckpt_cov-7000 --lr=0.001
 ```

#### Decode(output final summary)
 ```shell script
python main.py --model=end2end --mode=evalall --data_path=data/IAEA/finished_files_twitter/chunked/test_* --vocab_path=data/IAEA/finished_files_twitter/vocab --log_root=log_gpu_endlr0001 --exp_name=exp_sample --max_enc_steps=800 --max_dec_steps=120 --use_temporal_attention=True --intradecoder=True --eta=2.5E-05 --max_art_len=110 --max_sent_len=50 --decode_method=beam --coverage=True --single_pass=1 --save_pkl=True --save_vis=False --inconsistent_loss=True --inconsistent_topk=3 --eval_method=loss --load_best_eval_model=False --coverage=True --rl_training=True --eval_ckpt_path=log_gpu_endlr0001/end2end/exp_sample/train/model.ckpt_cov-10000
 ```

## Aknowledgement
The code of IAEA-Model is modified on the basis of [unified-summarization](https://github.com/HsuWanTing/unified-summarization) and [RLSeq2Seq](https://github.com/yaserkl/RLSeq2Seq).
