import unittest
import torch as t
import pandas as pd
import numpy as np
import os

import warnings
warnings.simplefilter("ignore")

ID = 4  # identifier for dispatcher

class TestDataset(unittest.TestCase):

    def setUp(self):
        # locate the csv file in file system and read it
        csv_path = ''
        for root, _, files in os.walk('.'):
            for name in files:
                if name == 'data.csv':
                    csv_path = os.path.join(root, name)
        self.assertNotEqual(csv_path, '', 'Could not locate the data.csv file')
        self.tab = pd.read_csv(csv_path, sep=';')

    def test_shape(self):
        from data import ChallengeDataset
        temp = ChallengeDataset(self.tab, 'val')
        val_dl = t.utils.data.DataLoader(temp, batch_size=1)
        for x, y in val_dl:
            x = x[0].cpu().numpy()
            self.assertEqual(x.shape[0], 3, 'Make sure that your images are converted to RGB')
            self.assertEqual(x.shape[1], 300, 'Your samples are not correctly shaped')
            self.assertEqual(x.shape[2], 300, 'Your samples are not correctly shaped')

            y = y[0].cpu().numpy()
            self.assertEqual(y.size, 2)

            break

    def test_normalization(self):
        from data import ChallengeDataset

        val_dl = t.utils.data.DataLoader(ChallengeDataset(self.tab, 'val'), batch_size=1)
        a = 0.0
        s = np.zeros(3)
        s2 = np.zeros(3)
        for x, _ in val_dl:
            x = x[0].cpu().numpy()
            a += np.prod(x.shape[1:])
            s += np.sum(x, axis=(1, 2))
            s2 += np.sum(x ** 2, axis=(1, 2))

        for i in range(3):
            self.assertTrue(-a * 0.09 < s[i] < a * 0.09, 'Your normalization seems wrong')
            self.assertTrue(a * 0.91 < s2[i] < a * 1.09, 'Your normalization seems wrong')


class TestModel(unittest.TestCase):
    def setUp(self):
        from trainer import Trainer
        from model import ResNet

        self.model = ResNet()
        crit = t.nn.BCELoss()
        trainer = Trainer(self.model, crit, cuda=False)
        trainer.save_onnx('checkpoint_test.onnx')

    def test_prediction(self):
        pred = self.model(t.rand((50, 3, 300, 300))) # batch, ch, ht, wt
        pred = pred.cpu().detach().numpy()

        self.assertEqual(pred.shape[0], 50)
        self.assertEqual(pred.shape[1], 2)
        self.assertFalse(np.isnan(pred).any(), 'Your prediction contains NaN values')
        self.assertFalse(np.isinf(pred).any(), 'Your prediction contains inf values')
        self.assertTrue(np.all([0 <= pred, pred <= 1]), 'Make sure your predictions are sigmoided')

    def test_prediction_after_save_and_load(self):
        import onnxruntime

        ort_session = onnxruntime.InferenceSession('checkpoint_test.onnx')
        ort_inputs = {ort_session.get_inputs()[0].name: t.rand((50, 3, 300, 300)).numpy()}
        pred = ort_session.run(None, ort_inputs)[0]

        self.assertEqual(pred.shape[0], 50)
        self.assertEqual(pred.shape[1], 2)
        self.assertFalse(np.isnan(pred).any(), 'Your prediction contains NaN values')
        self.assertFalse(np.isinf(pred).any(), 'Your prediction contains inf values')
        self.assertTrue(np.all([0 <= pred, pred <= 1]), 'Make sure your predictions are sigmoided')
