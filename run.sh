#!/usr/bin/bash

python -c "import logging; logging.basicConfig(level=logging.INFO); import data; data.main()"
python scripts/get_citations_and_references.py

for i in 1 2 3 4 5; do
  python scripts/predict_gender.py -i $i
done

python scripts/embed_titles.py

# The is the main experimental script that takes all of the data generated in the prior
# steps and returns the main analysis outputs. If you are interested in simply replicating
# our reported results without collecting the raw data yourself via multiple API calls and
# LLM queries, you can just run the following python script, replacing the -s argument
# with your desired medical area of interest. Run
#
# python scripts/run_analysis.py --help
#
# for more details.
for subject in \
  "allergy" \
  "cardiology" \
  "criticalcare" \
  "endocrinology" \
  "gastroenterology" \
  "geriatrics" \
  "infectiousdisease" \
  "medicine" \
  "nephrology" \
  "oncology" \
  "primarycare" \
  "pulmonology" \
  "rheumatology" \
; do
  echo $subject
  python scripts/run_analysis.py -m llm -s $subject
done

python scripts/build_gender_df.py