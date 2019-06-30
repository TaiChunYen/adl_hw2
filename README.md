# ELMO enhance sequence classification
![image](https://github.com/TaiChunYen/adl_hw2/blob/master/picture/task.jpg)
## NLP model
![image](https://github.com/TaiChunYen/adl_hw2/blob/master/picture/nlp_model.jpg)
## train ELMO
![image](https://github.com/TaiChunYen/adl_hw2/blob/master/picture/elmo_train.jpg)  
<br/>
<br/>
reproduce detail in ELMO/README.md

## train and predict BCN model
![image](https://github.com/TaiChunYen/adl_hw2/blob/master/picture/bcn_result.jpg)  
<br/>
<br/>
reproduce detail in BCN/README.md

## relate:BERT
convert csv to tsv:python csv2tsv.py
--csv2tsv.py  
<br/>
<br/>
### run self classification task:
bert/
--run_classifier.py
|--必要：
| |--data_dir  
| |--bert_config_file  
| |--task_name  
| |--vocab_file  
| |--output_dir  
|--其他設定：
&nbsp; |--init_checkpoint  
&nbsp; |--do_lower_case  
&nbsp; |--max_seq_length  
&nbsp; |--do_train  
&nbsp; |--do_eval  
&nbsp; |--do_predict  
&nbsp; |--train_batch_size  
&nbsp; |--eval_batch_size  
&nbsp; |--predict_batch_size  
&nbsp; |-learning_rate  
&nbsp; |-num_train_epochs  
&nbsp; |-warmup_proportion  
&nbsp; |-save_checkpoints_steps  
&nbsp; |-iterations_per_loop  
&nbsp; |-use_tpu  
<br/>
<br/>

### make predict csv:
python bert_process.py
--bert_process.py  
--sample_submission.csv  






