# crop-type-mapping

Crop type mapping of small holder farms in Ghana and South Sudan

##### INSTALLATION INSTRUCTIONS #####

Install Python 3.6

Install conda and build the environment with the following command:

`conda env create -f environment.yaml`

##### DATASET / ENVIRONMENT SETUP #####

These datasets are now available for free on Radiant Earth's MLHub, and through Sustain Bench.

###### Radiant Earth MLHub
- The dataset for Ghana is here: http://registry.mlhub.earth/10.34911/rdnt.ry138p/
- The dataset for South Sudan is here: http://registry.mlhub.earth/10.34911/rdnt.v6kx6n/
- The dataset files are saved as tifs, and will need to be restructured to work as input to the model, which initially used an hdf5 file. 

###### Sustain Bench
- See more information here: https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_type_mapping_ghana-ss.html

##### RUN INSTRUCTIONS #####

To visualize training, open a separate terminal and run the following before running the main training code:

  `python -m visdom.server`

Replace “localhost” with the static IP address provided on google cloud

To start training models, use the train.py script in the root directory of the code. 

Example for CLSTM-only network:
```python train.py --model_name=only_clstm_mi --country=southsudan --var_length --name=southsudan_clstmonly --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.003 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True```

Example for 3D UNet model: 
```python train.py --model_name=unet3d --country=southsudan --num_timesteps=24 --lr=0.0003 --s2_agg=False --include_indices=True --include_doy=True --use_planet=True --planet_agg=False --name=southsudan_3dunet_use_planet_noagg --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True --hidden_dims=128```

Example for Multi-Input 2D UNet + CLSTM model:
`python train.py --model_name=mi_clstm --country=ghana --var_length --main_crnn=True --early_feats True --include_indices=True --include_doy False --sample_w_clouds False --include_clouds False --lst_cloudy False --use_planet=True --resize_planet=False --name=ghana_use_planet_highres --env_name=ghana_use_planet_highres --use_s1=True --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.003 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True`

Example for Multi-Input 2D UNet + CLSTM earlier fused model:
```python train.py --model_name=fcn_crnn --country=southsudan --name=southsudan_fcn_crnn --env_name=myenv --dataset=full --epochs=130 --batch_size=5 --optimizer=adam --lr=0.001 --weight_decay=0 --loss_weight=True --weight_scale=1 --seed=1 --s2_num_bands=10 --dropout=0.5 --clip_val=True --include_s1=True```

Model type and country are set with the `model_name` and `country` flags, respectively. Flags also control several properties of inputs such as satellite types, temporal aggregation, Planet imagery resolution, cloud sampling, etc. Additional hyperparameter tuning settings can be set by invoking the appropriate flags. See `get_train_parser` in `util.py` for full details. 

