# Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs

## Paper Link

https://aclanthology.org/2023.emnlp-main.745.pdf

## Resources

Data links can be found in Experiment section of the paper.

Download bert_base_uncased from https://huggingface.co/bert-base-uncased

## Environment SetUp

pip install -r requirements.txt

## Augmentation through Cross-Sentence Pairs Extraction

Go inside the folder cross_sentence_pairs_extraction

With a list of raw sentences, consecutively run extract_templates.py --> nli_graph_construct.py --> pairs_generation.py to get cross-sentence pairs. 

If the raw sentences contain entity type, such as "And strangely enough , <e2:PERSON> Cain </e2:PERSON> 's short , three-year tenure at the <e1:ORGANIZATION> NRA </e1:ORGANIZATION> is evidently the only period in his decades-long career during which he 's alleged to have been a sexual predator .", run the "typed" version codes instead.

## Train & Evaluate

Once you prepared all required data files, you can train the model by running

python main_AugURE.py --temperature 0.02 --mlp --aug-plus --cos --dist-url tcp://localhost:10000 --multiprocessing-distributed --world-size 1 --rank 0 your_data_folder/  

To evaluate, run

python eval_cluster.py --temperature 0.02 --mlp --aug-plus --cos --dist-url tcp://localhost:10000 --multiprocessing-distributed --world-size 1 --rank 0 your_data_folder/





