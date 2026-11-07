"""Synthetic-data checks for training, checkpointing and held-out evaluation."""
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "experiments"))
from data_utils import SliceDataset, read_manifest, mix_liver_window_views
from main_experiment import reconstruction_loss, kd_loss


def write_data(tmp_path):
    rows=[]
    rng=np.random.default_rng(4)
    for patient, part, count in [("a","train",2),("b","val",1),("c","test",1)]:
        for i in range(count):
            path=tmp_path/f"{patient}_{i}.npz"
            np.savez(path,image=rng.random((16,16),dtype=np.float32),lower_hu=-1024.,width_hu=3024.)
            rows.append({"patient_id":patient,"partition":part,"image_path":path.name})
    manifest=tmp_path/"slices.csv"
    with manifest.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    return manifest,rows


def test_patient_isolation(tmp_path):
    manifest,rows=write_data(tmp_path)
    assert len(SliceDataset(manifest,"train",16,True))==2
    rows[-1]["patient_id"]="a"
    with manifest.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="multiple partitions"):
        read_manifest(manifest)
    with pytest.raises(ValueError,match="train partition"):
        SliceDataset(manifest,"test",16,True)


def test_weighted_loss_and_kd_mask():
    pred=torch.tensor([1.,3.]).reshape(2,1,1,1).requires_grad_()
    target=torch.zeros_like(pred)
    domain=torch.tensor([0.,1.]).reshape(2,1,1,1)
    loss=reconstruction_loss({"sr":pred,"preliminary_sr":pred},target,domain,0.)
    torch.testing.assert_close(loss,torch.tensor(5./3.))
    kd=kd_loss(pred,target,domain)
    torch.testing.assert_close(kd,torch.tensor(1.))
    kd.backward()
    assert pred.grad[1].item()==0
    assert kd_loss(pred,target,torch.ones_like(domain)).item()==0


def test_window_calibration():
    image=torch.tensor([0.,.5,1.]).reshape(1,1,1,3)
    mixed,domain=mix_liver_window_views(image,1.,[-20.],[120.])
    torch.testing.assert_close(mixed,image)
    assert domain.item()==1
    with pytest.raises(ValueError,match="calibration|requires"):
        mix_liver_window_views(image,1.,[0.],[0.])


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True,timeout=120)


def test_teacher_student_cli(tmp_path):
    manifest,_=write_data(tmp_path)
    cfg=yaml.safe_load((ROOT/"configs/teacher.yaml").read_text())
    cfg["device"]="cpu"
    cfg["model"].update(channels=8,depth=4,exchange_positions=[2])
    cfg["data"].update(manifest=str(manifest),patch_size=16)
    cfg["train"].update(batch_size=2,workers=0,max_steps=2,warmup_steps=1,
                        hold_until_step=1,validate_every=1,log_every=1)
    cfg["output_dir"]=str(tmp_path/"teacher")
    teacher_config=tmp_path/"teacher.yaml"
    teacher_config.write_text(yaml.safe_dump(cfg))
    run("experiments/main_experiment.py","--config",teacher_config)
    cfg["output_dir"]=str(tmp_path/"student")
    cfg["train"].update(kd_weight=.2,window_probability=.3,
                        teacher_checkpoint=str(tmp_path/"teacher/best.pt"))
    student_config=tmp_path/"student.yaml"
    student_config.write_text(yaml.safe_dump(cfg))
    run("experiments/main_experiment.py","--config",student_config)
    run("experiments/main_experiment.py","--config",student_config,"--resume")
    report=tmp_path/"test.json"
    run("experiments/evaluate.py","--checkpoint",tmp_path/"student/best.pt",
        "--manifest",manifest,"--output",report,"--device","cpu")
    result=json.loads(report.read_text())
    assert result["partition"]=="test" and result["patients"]==1
    assert result["data_consistency"]<2e-6
    run("experiments/evaluate.py","--checkpoint",tmp_path/"student/best.pt",
        "--input",tmp_path/"c_0.npz","--output",tmp_path/"sr.npy","--device","cpu")
    assert np.load(tmp_path/"sr.npy").shape==(32,32)
