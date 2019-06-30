# ELMo
## create embedding.pkl and word_to_index:
python make_dataset.py create/data/save/direction/
--make_dataset.py
--preprocessor.py
--embedding.py

## train ELMo:
python train_elmo.py
--train_elmo.py
--elmo_for_train.py

##use on BCN:
--embedder.py
--elmo_wo_cnn.py


