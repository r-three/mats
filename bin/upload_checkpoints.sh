# for checkpoint in exp_out/p3/cosmos_qa/google-t5-large-lm-adapt/2023-04-30-10-16-25/checkpoints/checkpoint_399.pt \
# exp_out/p3/paws/google-t5-large-lm-adapt/2023-04-30-00-43-21/checkpoints/checkpoint_1299.pt \
# exp_out/p3/qasc/google-t5-large-lm-adapt/2023-04-29-21-15-10/checkpoints/checkpoint_199.pt \
# exp_out/p3/quail/google-t5-large-lm-adapt/2023-04-29-21-11-24/checkpoints/checkpoint_999.pt \
# exp_out/p3/quartz/google-t5-large-lm-adapt/2023-04-29-21-13-32/checkpoints/checkpoint_399.pt \
# exp_out/p3/ropes/google-t5-large-lm-adapt/2023-04-29-21-11-51/checkpoints/checkpoint_499.pt \
# exp_out/p3/social_iqa/google-t5-large-lm-adapt/2023-04-29-21-13-12/checkpoints/checkpoint_799.pt \
# exp_out/p3/wiki_qa/google-t5-large-lm-adapt/2023-04-29-21-15-46/checkpoints/checkpoint_499.pt \
# exp_out/p3/cosmos_qa/google-t5-large-lm-adapt/ia3/2023-05-01-10-25-08/checkpoints/checkpoint_1299.pt \
# exp_out/p3/paws/google-t5-large-lm-adapt/ia3/2023-04-30-19-54-04/checkpoints/checkpoint_2699.pt \
# exp_out/p3/qasc/google-t5-large-lm-adapt/ia3/2023-05-01-10-23-07/checkpoints/checkpoint_499.pt \
# exp_out/p3/quail/google-t5-large-lm-adapt/ia3/2023-04-30-18-25-27/checkpoints/checkpoint_1399.pt \
# exp_out/p3/quartz/google-t5-large-lm-adapt/ia3/2023-04-30-19-53-11/checkpoints/checkpoint_1799.pt \
# exp_out/p3/ropes/google-t5-large-lm-adapt/ia3/2023-04-30-18-26-14/checkpoints/checkpoint_799.pt \
# exp_out/p3/social_iqa/google-t5-large-lm-adapt/ia3/2023-04-30-18-26-30/checkpoints/checkpoint_1099.pt \
# exp_out/p3/wiki_qa/google-t5-large-lm-adapt/ia3/2023-04-30-23-57-20/checkpoints/checkpoint_99.pt 
# for checkpoint in exp_out/p3_eight_qa/p3/google-t5-large-lm-adapt/full_model/2023-09-25-04-32-13/checkpoints/checkpoint_1999.pt \
# exp_out/p3_eight_qa/p3/google-t5-large-lm-adapt/ia3/2023-06-18-01-43-24/checkpoints/checkpoint_9999.pt
# do
#     exp_dir=$(dirname $(dirname $checkpoint))
#     gsutil cp "${exp_dir}/*.json" "gs://merging_by_matching_models_in_their_task_subspace/${exp_dir}"
#     gsutil cp  "${checkpoint}"  "gs://merging_by_matching_models_in_their_task_subspace/${checkpoint}"
# done


for checkpoint in exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/average/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/task_vectors/model_lambda_0.3/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/ties/model_lambda_1.0/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/fisher_merging/diagonal_empirical_fisher_validation/model_lambda_1.0/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/regmean/validation/model_lambda_0.8/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/conjugate_gradients/regmean_validation_initialize_task_vectors_model_lambda_1.0_iterations_20/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/average/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/task_vectors/model_lambda_0.2/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/ties/model_lambda_1.0/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/fisher_merging/diagonal_empirical_fisher_validation/model_lambda_1.0/merged_model.pt \
exp_out/old_merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/conjugate_gradients/fisher_merging_blockwise_empirical_fisher_validation_initialize_task_vectors_model_lambda_1.0_iterations_10/merged_model.pt
do
    new_checkpoint="${checkpoint/old_merging/merging}"
    gsutil cp  "${checkpoint}"  "gs://merging_by_matching_models_in_their_task_subspace/${new_checkpoint}"
done


