import pytest
import numpy as np
import vivae as vv

def test_res():
    np.random.seed(1)
    x = np.random.rand(50, 2).astype(np.float32)
    
    model = vv.ViVAE(input_dim=x.shape[1], latent_dim=2)
    model.fit(
        x, n_epochs=1, lam_recon=1., lam_kldiv=.1, lam_geom=1., lam_egeom=1., lam_mds=1., batch_size=2
    )
    emb = model.transform(x)
    assert emb.shape[0]==x.shape[0]
    assert emb.shape[1]==2
