DATASET_DIR=/home/lab/dataset/cancer_dataset/train2_processed/
LABEL=/home/lab/dataset/cancer_dataset/train2_label.csv
VAL_LABEL=/home/lab/dataset/cancer_dataset/train_label.csv
MODEL_SAVE_DIR=./output_train1
python train.py \
	--learning_rate=1e-4 \
	--learning_rate_fine=1e-5 \
	--max_steps=5000 \
	--batch_size=6 \
	--dataset_dir = ${DATASET_DIR} \
	--model_save_dir=${MODEL_SAVE_DIR} \
	--use_pretrain_model=False \
	--gpu_num=2 \
	--train_list=${LABEL} \
	--test_list=${VAL_LABEL} \
	--MOVING_AVERAGE_DECAY=0.9995


