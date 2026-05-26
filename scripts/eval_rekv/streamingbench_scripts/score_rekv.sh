cd ../src/data

# python count.py --model "<model_name>" --task "<real/omni/sqa/proactive>" --src "<output_file>"

python src/data/count.py --model "rekv" --task "real" --src "/src/data.json"
