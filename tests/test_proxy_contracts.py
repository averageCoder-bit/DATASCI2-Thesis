"""Read-only source-contract tests for LGMMA-X proxy labels.
Run: python -m unittest discover -s tests -p test_proxy_contracts.py -v
This file is intended to be copied to the repository's tests/ directory.
"""
from __future__ import annotations
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROXY = REPO / 'src/modeling/evaluation/proxy_labels.py'


def load_proxy():
    spec = importlib.util.spec_from_file_location('lgmma_proxy_under_test', PROXY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProxyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy = load_proxy()

    def test_extreme_return_std_is_sample_ddof_one(self):
        values = np.array([.001,.002,.003,.001], dtype=float)
        df = pd.DataFrame({
            'log_return':values,
            'range_pct':[.01,.015,.020,.025],
            'rolling_volatility':[.001,.002,.003,.004],
            'shock':[1.,2.,3.,4.],
        })
        actual = self.proxy.derive_thresholds(df)['extreme_return']
        expected = float(values.mean()+3*values.std(ddof=1))
        self.assertAlmostEqual(actual, expected, places=13)
        self.assertNotAlmostEqual(actual,float(values.mean()+3*values.std(ddof=0)),places=8)

    def test_flash_reversal_and_final_valid_start(self):
        # Seven closes, i=0 is last valid 2+4 forward-look position.
        df = pd.DataFrame({'segment_id':[0]*7,
                           'Close':[3000,2990,2970,2974,2979,2984,2988]})
        got = self.proxy.calculate_flash_movement(df)
        self.assertEqual(got.loc[0,'shock'],30.0)
        self.assertEqual(got.loc[0,'reversal'],18.0)
        self.assertAlmostEqual(got.loc[0,'retracement_ratio'],.6)
        self.assertTrue(np.isnan(got.loc[1,'shock']))
        # The next segment must never be consulted as a continuation.
        short = pd.DataFrame({'segment_id':[1]*6+[2]*2,
                              'Close':[3000,2990,2970,2974,2979,2984,2988,3000]})
        got = self.proxy.calculate_flash_movement(short)
        self.assertTrue(got['shock'].isna().all())

    def test_flash_requires_shock_and_retracement(self):
        df = pd.DataFrame({'shock':[30.,19.,30.,30.],
                           'retracement_ratio':[.6,.6,.4,np.nan],
                           'log_return':[0]*4,'range_pct':[0]*4,
                           'rolling_volatility':[0]*4})
        thresholds = {'extreme_return':1.,'range_expansion':1.,
                      'volatility_burst':1.,'flash_shock':20.}
        labels = self.proxy.apply_timestep_labels(df, thresholds)
        self.assertEqual(labels['flash_movement'].tolist(),[True,False,False,False])

    def test_single_flag_makes_sixty_candle_sequence_positive(self):
        dates = pd.date_range('2026-01-05',periods=60,freq='5min',tz='UTC')
        d = pd.DataFrame({'segment_id':[0]*60,'Date':dates,
                          'extreme_return':[False]*60,
                          'range_expansion':[False]*60,
                          'volatility_burst':[False]*60,
                          'flash_movement':[False]*60,
                          'proxy_anomaly_timestep':[False]*60})
        d.loc[37,'flash_movement']=True
        d.loc[37,'proxy_anomaly_timestep']=True
        md = pd.DataFrame({'sequence_id':[0],'segment_id':[0],
                          'start_timestamp':[dates[0]],'end_timestamp':[dates[-1]]})
        result = self.proxy.aggregate_sequence_labels(d,md)
        self.assertEqual(int(result.loc[0,'proxy_anomaly']),1)
        self.assertEqual(int(result.loc[0,'anomalous_timestep_count']),1)
        self.assertEqual(int(result.loc[0,'flash_movement']),1)

if __name__ == '__main__':
    unittest.main()
