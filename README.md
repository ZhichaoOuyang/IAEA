# IAEA
IAEA (Integrity-Aware Extractive-Abstractive realtime event summarization) is an unified extractive-abstractive framework for realtime event summarization. Our key idea is to integrate an inconsistency detection module to preserve the integrity of the summaries in each time slice.

## HID_model
### environment
python3
tensorflow 1.13.1
keras 2.4.2

run **python HDLTweet_nopos.py**


## IAEA_model
### environment
python3
tensorflow 1.2.1-gpu

### data preprocess
run **python data/IAEA/make_datafiles_testdata.py twitter_final/train** to get training data
to same to valid and test data.

### Extractor
run **python main_gpu.py --model=selector --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_gpu_8m6_selector001 --exp_name=exp_sample --max_art_len=110 --max_sent_len=50 --max_train_iter=1500 --batch_size=5 --save_model_every=500 --lr=0.01 --model_max_to_keep=25**

### Abstractor
run **python main_gpu3_8m10.py --model=rewriter --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_rewriter_8m6 --exp_name=exp_sample --max_enc_steps=400 --max_dec_steps=100 --batch_size=5 --max_train_iter=5000 --save_model_every=1000 --model_max_to_keep=5**

**python main_gpu.py --model=rewriter --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_rewriter_8m6 --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True  --convert_to_reinforce_model=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10**

**python main_gpu.py --model=rewriter --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_rewriter_8m6 --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10**

**python main_gpu.py  --model=rewriter --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_rewriter_8m12 --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10 --coverage=True --convert_to_coverage_model=True**

**python main_gpu_8m10.py  --model=rewriter --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_rewriter_8m12 --exp_name=exp_sample --batch_size=5 --max_train_iter=1000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --max_enc_steps=400 --max_dec_steps=100 --save_model_every=100 --model_max_to_keep=10 --coverage=True**

### end2end
**python main_gpu.py --model=end2end --mode=train --data_path=data/finished_files_twitter_8m6/chunked/train_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_gpu_endlr0001 --exp_name=exp_sample --max_enc_steps=800 --max_dec_steps=120 --max_train_iter=10000 --batch_size=5 --use_temporal_attention=True --intradecoder=True --eta=2.5E-05 --max_art_len=110 --max_sent_len=50 --selector_loss_wt=5.0 --inconsistent_loss=True --inconsistent_topk=3 --save_model_every=1000 --model_max_to_keep=20 --rl_training=True --coverage=True --pretrained_selector_path=log_gpu_8m6_selector001/selector/exp_sample/train/model.ckpt-500 --pretrained_rewriter_path=log_rewriter_8m12/rewriter/exp_sample/train/model.ckpt_cov-7000 --lr=0.001**

#### decode
**python main_gpu.py --model=end2end --mode=evalall --data_path=data/finished_files_twitter_8m6/chunked/test_* --vocab_path=data/finished_files_twitter_8m6/vocab --log_root=log_gpu_endlr0001 --exp_name=exp_sample --max_enc_steps=800 --max_dec_steps=120 --use_temporal_attention=True --intradecoder=True --eta=2.5E-05 --max_art_len=110 --max_sent_len=50 --decode_method=beam --coverage=True --single_pass=1 --save_pkl=True --save_vis=False --inconsistent_loss=True --inconsistent_topk=3 --eval_method=loss --load_best_eval_model=False --coverage=True --rl_training=True --eval_ckpt_path=log_gpu_endlr0001/end2end/exp_sample/train/model.ckpt_cov-10000**


