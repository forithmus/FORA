"""CPU tests of the X-ray location text rules and the leave-one-out location statistics."""

import numpy as np
import torch

from fora.reportgen.cxr import location as LOC
from fora_report.cxr.data import epoch_schedule
from fora.reportgen.cxr.stream import location_statistics


def test_describe_rules():
    assert LOC.describe('none', 1.0, 0.9, [1, 0, 0]) is None
    assert LOC.describe('side_zone', 0.01, 0.9, [1, 0, 0]) is None
    assert LOC.describe('side_zone', 1.0, 0.9, [0.1, 0.2, 0.7]) == 'right lower zone'
    assert LOC.describe('side_zone', 1.0, 0.2, [0.4, 0.3, 0.3]) == 'left upper and mid zones'
    assert LOC.describe('side_zone', 1.0, 0.5, [0.1, 0.2, 0.7]) == 'bilateral, mainly lower zone'
    assert LOC.describe('side_zone', 1.0, 0.5, [0.29, 0.29, 0.29]) == 'bilateral, all zones'
    assert LOC.describe('side', 1.0, 0.75, [1, 0, 0]) == 'right side'
    st = LOC.style_table(['Pleural effusion', 'Hilar enlargement', 'Cardiomegaly'],
                         {'Pleural effusion': 'Pleura', 'Hilar enlargement': 'Mediastinum & hila',
                          'Cardiomegaly': 'Cardiac & pericardial'})
    assert st == {'Pleural effusion': 'side_zone', 'Hilar enlargement': 'side', 'Cardiomegaly': 'none'}


def test_location_store(tmp_path):
    classes = ['Pleural effusion', 'Cardiomegaly']
    np.savez(tmp_path / 'loc.npz', ids=np.array(['s1']), classes=np.array(classes),
             mass=np.array([[1.0, 1.0]], np.float16), frac_right=np.array([[0.1, 0.9]], np.float16),
             zone=np.array([[[0, 0, 1], [1, 0, 0]]], np.float16))
    store = LOC.LocationStore(str(tmp_path / 'loc.npz'), classes, {'Pleural effusion': 'Pleura'})
    assert store.texts('s1') == {0: 'left lower zone'}


def test_location_statistics_side_and_zone():
    # one frontal film, 2x2 grid; all evidence of class 0 on token (row 1, col 0) = image left = patient right, lower
    row = torch.tensor([0., 0., 1., 1.]); col = torch.tensor([0., 1., 0., 1.])
    gy = torch.full((4,), 2.); gx = torch.full((4,), 2.)
    L = torch.tensor([[0.], [0.], [5.], [0.]])
    w = torch.full((4, 1), 0.25)
    out = location_statistics(L, w, torch.ones(1), (row, col, gy, gx), torch.zeros(4, dtype=torch.long))
    mass, fr, up, mid, low = out[0]
    assert mass > 0 and abs(fr - 1.0) < 1e-6 and low > 0.99 and up == 0 and mid == 0
    lateral = location_statistics(L, w, torch.ones(1), (row, col, gy, gx), torch.ones(4, dtype=torch.long))
    assert lateral[0, 0] == 0


def test_epoch_schedule_covers_all_items():
    sizes = np.arange(37)
    s = epoch_schedule(37, sizes, 4, 0, 42, bucket=2)
    assert s.shape == (10, 4) and set(s.ravel().tolist()) == set(range(37))
    assert (s == epoch_schedule(37, sizes, 4, 0, 42, bucket=2)).all()


def test_stream_prepare_and_geometry():
    import numpy as np
    import torch

    from fora.reportgen.cxr.stream import CXRVisualStream
    s = CXRVisualStream.__new__(CXRVisualStream)
    s.cfg, s.device = {'max_films': 8, 'tile': 1024, 'position_freqs': 4}, torch.device('cpu')
    case = {'id': 'x', 'films': [['a', 'a', 0, 0, 0], ['b', 'b', 1, 0, 0], ['c', 'c', 2, 0, 0]]}

    def load_film(film):
        if film[0] == 'c':
            raise IOError('unreadable')
        return np.zeros((300, 1100) if film[0] == 'a' else (256, 256), np.float32)
    bag = s.prepare(case, load_film=load_film)
    assert bag['grids'] == [(19, 69), (16, 16)] and int(bag['kind'].numel()) == 19 * 69 + 256
    row, col, gy, gx = s.film_geometry(bag)
    assert len(row) == 19 * 69 + 256 and int(col.max()) == 68 and int(gy[-1]) == 16
