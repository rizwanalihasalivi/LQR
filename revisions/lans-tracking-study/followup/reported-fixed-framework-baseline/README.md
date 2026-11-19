# Retained LANS reference

This ranking was supplied by the user for the `lans-fixed-framework` experiment. LANS placed first in 8 of 14 conditions, second in 6, and completed 420 of 420 episodes. PPO is the frozen equal-pretrained PPO, and MPC is objective-matched MPC.

The selected policy is RMSE-aligned checkpoint 20 with bounded PPO refinement (post-PPO), for policy seeds 272, 273, and 274. Evaluation seeds were 241000–241004. The reported source is `revisions/lans-fixed-framework/Test_Report.md`.

The six targets are the four reference families at 1650 kg and the DLC runs at 1100 and 2200 kg, all without delay. Evaluate any candidate across all 14 conditions with the same frozen baselines and check the eight existing first-place ranks and the episode completion count.
