#!/bin/bash
#SBATCH --job-name=cwqp-7B
#SBATCH --output=[1007-FIN2-top10]CWQ-7b-b.txt

SPLIT="test"
DATASET_LIST="RoG-cwq"
MODEL_NAME=FIN2-CWQ-llama-7-b-top10
PROMPT_PATH=prompts/llama2_predict.txt
BEAM_LIST="3" # "1 2 3 4 5"

# GNN-RAG-RA
time {
for DATA_NAME in $DATASET_LIST; do
    for N_BEAM in $BEAM_LIST; do
        RULE_PATH=/share0/khyun33/myExp/infer-cwq/test_simple_add_path_GCR_acc.jsonl
        RULE_PATH_G1=llm/results/gnn/${DATA_NAME}/rearev-sbert/_test.info
        RULE_PATH_G2=None #llm/results/gnn/${DATA_NAME}/rearev-lmsr/test.info

        python -u src/qa_prediction/predict_answer.py \
            --model_name ${MODEL_NAME} \
            -d ${DATA_NAME} \
            --prompt_path ${PROMPT_PATH} \
            --add_rule \
            --rule_path ${RULE_PATH} \
            --rule_path_g1 ${RULE_PATH_G1} \
            --rule_path_g2 ${RULE_PATH_G2} \
            --model_path /share0/yjko1853/myExp/[STEP2]friendly-sft/JOINT/final/cwq-noise-no-literal/final_step_webqsp/7B-SIMPO-CWQ-WEBQSP/lora_model \
            --predict_path results/KGQA-GNN-RAG-RA/rearev-sbert
            
    done
done
}