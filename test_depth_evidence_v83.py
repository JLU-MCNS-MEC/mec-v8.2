"""CPU-only sensor, normalization and no-oracle Depth feature tests."""
import unittest
import numpy as np
import torch
from depth_evidence_v83 import goal_conditioned_depth_evidence
from config import ENCODER, AUX

class DepthEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.current=torch.zeros(1,ENCODER.num_tokens,ENCODER.token_dim)
        self.current[...,0]=1
        self.goal=torch.zeros(1,AUX.goal_token_count,ENCODER.token_dim)
        self.goal[...,0]=1

    def test_missing_returns_are_not_near(self):
        a=goal_conditioned_depth_evidence(self.current,self.goal,np.zeros((336,336),dtype=np.float32))
        self.assertEqual(tuple(a.shape),(1,8))
        self.assertEqual(float(a[0,3]),0.0)
        self.assertEqual(float(a[0,4]),0.0)
        self.assertTrue(torch.isfinite(a).all())

    def test_constant_depth_is_normalized_slant_range(self):
        a=goal_conditioned_depth_evidence(self.current,self.goal,np.full((336,336),125.0,dtype=np.float32))
        self.assertAlmostEqual(float(a[0,0]),125.0/float(AUX.depth_max_m),places=4)
        self.assertAlmostEqual(float(a[0,3]),1.0,places=4)
        self.assertAlmostEqual(float(a[0,4]),1.0-125.0/float(AUX.depth_max_m),places=4)
        self.assertTrue(((a>=0)&(a<=1)).all())

    def test_invalid_readings(self):
        a=goal_conditioned_depth_evidence(self.current,self.goal,torch.full((336,336),float('nan')))
        self.assertTrue(torch.isfinite(a).all())
        self.assertAlmostEqual(float(a[0,3]),0.0)

    def test_reject_misaligned_depth(self):
        for bad in ((335,336),(336,337),(224,224),(336,224)):
            with self.assertRaises(ValueError):
                goal_conditioned_depth_evidence(self.current,self.goal,np.zeros(bad,dtype=np.float32))

    def test_real_sensor_numpy_interface(self):
        from env import ToyUAVEnv
        env=ToyUAVEnv()
        env.reset(seed=900001,render_observation=False)
        arr=env.render_depth()
        self.assertIsInstance(arr,np.ndarray)
        out=goal_conditioned_depth_evidence(self.current,self.goal,arr)
        self.assertEqual(tuple(out.shape),(1,8))
        self.assertTrue(torch.isfinite(out).all())

if __name__=="__main__":
    unittest.main(verbosity=2)
