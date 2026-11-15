import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from run import PALETTE, decode, metrics
from boundary import counts, self_test


class BaselineTest(unittest.TestCase):
    def test_metrics(self):
        self_test()
        labels=np.tile(np.arange(6,dtype=np.uint8),(6,1))
        scores=metrics(np.eye(6,dtype=np.int64)*6,counts(labels,labels))
        self.assertEqual(scores['mean_iou'],1)
        self.assertEqual(scores['boundary_f1'],1)
        with self.assertRaises(ValueError):
            decode(np.full((3,2,2),17,dtype=np.uint8),{'tile_id':'fake'})

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows=[]
            for i,split in enumerate(['train','validation','test']):
                labels=np.tile(np.arange(60,dtype=np.uint8)%6,(60,1))
                rgb=PALETTE[labels].transpose(2,0,1)
                for kind in ['image','mask']:
                    dest=root/'nested'/f'{i}_{kind}.tif'
                    dest.parent.mkdir(exist_ok=True)
                    with rasterio.open(dest,'w',driver='GTiff',height=60,width=60,count=3,dtype='uint8') as ds:
                        ds.write(rgb)
                rows.append(dict(tile_id=f'{i}_1',split=split,image=f'{i}_image.tif',mask=f'{i}_mask.tif'))
            (root/'manifest.json').write_text(json.dumps({'tiles':rows}))
            script=Path(__file__).with_name('run.py')
            subprocess.run([sys.executable,str(script),'train','--data',str(root),'--output',str(root/'train'),'--samples-per-tile','3000'],check=True,capture_output=True)
            search=json.loads((root/'train/search.json').read_text())
            self.assertEqual(len(search['trials']),4)
            subprocess.run([sys.executable,str(script),'evaluate','--data',str(root),'--output',str(root/'test'),'--checkpoint',str(root/'train/best.joblib'),'--split','test'],check=True,capture_output=True)
            report=json.loads((root/'test/report.json').read_text())
            self.assertEqual(report['prediction']['overall']['mean_iou'],1)
            self.assertEqual(report['prediction']['overall']['boundary_f1'],1)


if __name__=='__main__':
    unittest.main()
