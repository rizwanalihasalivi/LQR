"""Regression checks for delay coverage and per-minibatch PPO projection.

Run from any directory with the prepared Python interpreter. No checkpoints
or research results are written by these tests.
"""
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO
from features import NetworkEnv
from training_utils import ParameterBoxProjection

torch.set_num_threads(1)

class DelayCurriculumTests(unittest.TestCase):
 def test_actual_budget_reaches_full_delay_distribution(self):
  env=NetworkEnv(training=True,training_budget=4000)
  env.reset(seed=10)
  self.assertTrue(np.all(env.delays==0))
  env.total_steps=1200
  for seed in range(20):
   env.reset(seed=seed)
   self.assertTrue(np.isin(env.delays,[0.,10.]).all())
  env.total_steps=2400
  maximum=0.
  for seed in range(50):
   env.reset(seed=seed);maximum=max(maximum,float(env.delays.max()))
  self.assertEqual(maximum,200.)
  env.close()

 def test_pretrained_finetuning_exposes_high_delay_at_start(self):
  env=NetworkEnv(training=True,training_budget=100352,training_delay_curriculum=False)
  for seed in range(50):
   env.reset(seed=seed)
   env.step([0.])
  self.assertGreater(env.training_delay_steps['above_10ms'],0)
  self.assertGreater(env.training_delay_steps['at_200ms'],0)
  self.assertEqual(env.training_delay_steps['maximum_ms'],200.)
  self.assertEqual(env.training_delay_steps['total'],50)
  self.assertEqual(env.training_budget,100352)
  env.close()

 def test_existing_constructor_keeps_default_curriculum(self):
  env=NetworkEnv(training=True)
  self.assertEqual(env.training_budget,200704)
  self.assertTrue(env.training_delay_curriculum)
  with self.assertRaises(ValueError):NetworkEnv(training_budget=0)
  env.close()

class ProjectionTests(unittest.TestCase):
 def test_every_actual_ppo_optimizer_step_is_bounded(self):
  env=NetworkEnv(training=True,training_delay_curriculum=False)
  model=PPO('MlpPolicy',env,n_steps=16,batch_size=8,n_epochs=2,
            learning_rate=.1,seed=42,device='cpu',verbose=0)
  for parameter in model.policy.mlp_extractor.policy_net.parameters():parameter.requires_grad_(False)
  model.policy.log_std.requires_grad_(False)
  projection=ParameterBoxProjection(model.policy.action_net.parameters(),1e-5)
  checks=[]
  project_hook=model.policy.optimizer.register_step_post_hook(projection.after_step)
  check_hook=model.policy.optimizer.register_step_post_hook(
   lambda optimizer,args,kwargs:checks.append(projection.max_change()))
  try:model.learn(32)
  finally:project_hook.remove();check_hook.remove();env.close()
  self.assertEqual(projection.optimizer_steps,8)
  self.assertEqual(len(checks),8)
  self.assertLessEqual(max(checks),1e-5+1e-7)
  self.assertGreater(max(checks),0.)

if __name__=='__main__':unittest.main(verbosity=2)
