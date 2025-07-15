# code_clone_dection
## example1: slide window + attention pooling
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2  main.py   --model_name_or_path microsoft/unixcoder-base   --num_train_epochs 5 --train_batch_size 16 --do_test  --do_train --do_eval   --output_dir unixcoderattention630 --train_filename dataset_new/pair_train.jsonl --dev_filename dataset_new/pair_valid.jsonl --test_filename dataset_new/pair_test.jsonl --max_source_length 1536 --pooling attention --slide_window_size 512  2>&1| tee train.log

## example2 baseline

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2  main.py   --model_name_or_path microsoft/unixcoder-base   --num_train_epochs 5 --train_batch_size 16 --do_test  --do_train --do_eval   --output_dir unixcoderbase630 --train_filename dataset_new/pair_train.jsonl --dev_filename dataset_new/pair_valid.jsonl --test_filename dataset_new/pair_test.jsonl --max_source_length 512 2>&1| tee train.log