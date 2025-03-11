
# install
* You can skip at the beggining.
if you have errors about those packages.

```
mkdir submodules
cd submodules
git clone --recursive https://github.com/hbb1/diff-surfel-rasterization.git
git clone --recursive https://gitlab.inria.fr/bkerbl/simple-knn.git

pip install ./diff-surfel-rasterization
pip install ./simple-knn
```

* You need to reate an account for weights and bias and log in before running.
  * https://wandb.ai/login
```
pip install wandb

wandb login
```

Please change the path.
* train.py
```
# TODO: please change this to your own folder
SHARED_FOLDER = "/home/hpc/iwi9/iwi9007h/Ref-NPR/matija"
```

# commands
```
# For tes run.
python train.py -s /home/hpc/iwi9/iwi9007h/Ref-NPR/matija/pollen --eval --test_iterations 60001 --checkpoint_iterations 60001 --iterations 60001 --train_color --start_checkpoint /home/hpc/iwi9/iwi9007h/Ref-NPR/matija/8ee5c65c-3/chkpnt60000.pth --ref_case pollen_color_artist_5_2dgs_1k --style_gt_weight 20.0 --tcm_weight 1.0 --online_tmp_weight 1.0 --patch_weight 0.0 --tv_weight 0 --depth_weight 0.0 --l1_loss_related_rays 1.0 --save_img_interval 1000 --lambda_dssim 0 --loss_names online_tmp_loss tcm_loss color_patch --gray_weight 0.00 --resolution 4

# for training.
python train.py -s /home/hpc/iwi9/iwi9007h/Ref-NPR/matija/pollen --eval --test_iterations 80000 --checkpoint_iterations 80000 --iterations 80000 --train_color --start_checkpoint /home/hpc/iwi9/iwi9007h/Ref-NPR/matija/8ee5c65c-3/chkpnt60000.pth --ref_case pollen_color_artist_5_2dgs_1k --style_gt_weight 20.0 --tcm_weight 1.0 --online_tmp_weight 1.0 --patch_weight 0.0 --tv_weight 0 --depth_weight 0.0 --l1_loss_related_rays 1.0 --save_img_interval 1000 --lambda_dssim 0 --loss_names online_tmp_loss tcm_loss color_patch --gray_weight 0.00 --resolution 4
```
