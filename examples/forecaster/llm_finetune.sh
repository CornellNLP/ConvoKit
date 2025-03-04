declare -a all_models=(
    "google/gemma-2-27b-it"
)
for corpus in "cga-cmv-large" "cga-wikiconv"
do
for model in "${all_models[@]}"; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python llm_forecaster.py\
        --model_name_or_path ${model}\
        --do_finetune True\
        --do_tune_threshold True\
        --corpus_name ${corpus}\
        --gradient_accumulation_steps 32\
        --learning_rate 1e-4\
        --per_device_batch_size 1\
        --num_train_epochs 1\
        --output_dir "/reef/sqt2/CGA_LLM/fine-tune/${corpus}/${model}"
done
done