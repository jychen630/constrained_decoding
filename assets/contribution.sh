#!/bin/bash
branches=("junyao" "junyao_fast_single_template_constraint" "pranithan/ordered_constraints" "raavi" "template_order_constraint")

output_file="commit_info.csv"
echo "Branch,Commit Hash,Author,Date,Title" > $output_file

for branch in "${branches[@]}"; do
    git checkout $branch
    git log --since="2025-04-09" --pretty=format:"$branch,%h,%an,%ad,%s" --date=iso >> $output_file
done
