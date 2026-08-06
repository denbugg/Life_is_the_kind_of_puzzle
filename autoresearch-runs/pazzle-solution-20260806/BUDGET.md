# Experiment budget

metric                 = mean_solve_ssim
direction              = higher
secondary_metrics      = mean_final_ssim, neighbour_accuracy, placement_accuracy, edge_r1
seed_experiments       = 6
seconds_per_experiment = 300
parallelism            = 1_gpu_or_3_cpu
compute_cap            = 2_gpu_hours
paid_cloud             = disabled

# Generational loop

max_generations        = 3
hypotheses_per_gen     = 3
proposers              = 3
critics                = 3
stagnation             = 2

# Hard gates

- No production integration without positive paired mean solve-only SSIM on the immutable 24-scene gate.
- A winner must repeat on a second predeclared corruption seed or an independent confirmation subset.
- No rotations: every board is a permutation of upright input tiles.
- Stop a two-side branch before packing if exact seed-motif precision is below 0.95 or unique seed-tile coverage is below 0.15.
- Stop launching GPU work at 120 GPU-minutes.

# Spent

generations_run = 1
experiments_run = 21
gpu_min_used    = approximately_19.8_research_plus_113.3_production_inference
best_metric     = budget96_repeated_positive_mean_on_two_24_scene_gates
champion        = raw_candidate_ranker_buddies_budget96_repair0_fixed_nlm_h10

# Generation 1 predeclaration

next_experiment = E22_higher_recall_singleton_emitter_candidate_ceiling_before_training
gate_v4         = E11_reject_final_delta_plus_0.000623_below_plus_0.001
keep_rule       = mean_final_delta_gt_0.001_and_mean_solve_delta_ge_0
fallback        = corruption_invariant_real_label_ranker_pilot
denoise_probe   = E12_killed_clean_oracle_solve_minus_0.007070_final_minus_0.016292
denoise_kill    = solve_delta_ge_0.010_final_delta_ge_0.015_wins_ge_6_of_8_worst_ge_minus_0.020
cc192_probe     = E14_reject_structure_passed_but_solve_minus_0.008794_final_minus_0.014433
future_storage  = E_drive_for_gate_v3_score_caches_and_v2_submission
rank96_v1_zip   = E_drive_complete_sha256_9a2eaf962507d11f2cad0caf59af40fe9755a6f092051c9d144a5f6aca10965f
external_score  = 0.2161981413457065_confirmed_for_rank96_v1
remote_training = not_launched_E12_and_E14_failed_end_to_end_transfer
frame_consensus = E15_killed_structure_three_true_relations_total_coverage_0.003689
restoration_probe = E16_predeclared_no_GPU_training_until_exact_clean_render_ceiling_passes
restoration_result = E16_killed_clean_render_minus_RR96_NLM_minus_0.015296_wins_1_of_8
rigid_viability = E17_pass_mean_pure_coverage_0.425781_worst_0.315972_added96_precision_0.930990
absolute_frame_beam = E18_killed_scene10_reached_fixed_500000_cap_before_any_candidate_board
relative_frame_beam = E19_killed_scene10_reached_fixed_500000_cap_after_32_rounds_with_one_zero_root
triangle_potential = E20_killed_all_quality_gates_mean_pose_coverage_0.036024_relation_precision_0.132524_cycle_ratio_zero
posegraph_candidate_ceiling = E21_killed_mean_exact_connected_coverage_0.039497_worst_0.019097_before_GPU_training
