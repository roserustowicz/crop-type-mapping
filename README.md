# crop-type-mapping
Crop type mapping of small holder farms in Ghana and South Sudan

##### INSTALLATION INSTRUCTIONS #####

Install Python 3.6

Install conda and build the environment with the following command:

`conda env create -f environment.yaml`

##### DATASET / ENVIRONMENT SETUP #####

To install and format data, follow instructions in environment_setup.txt

Data directories should be stored in the root /home folder. 

For example, the folder `/home/data/ghana/` should house all the split information, data.hdf5 files, original data files, etc. for ghana. 

##### RUN INSTRUCTIONS #####

To visualize training, open a separate terminal and run the following before running the main training code:

  `python -m visdom.server`

Replace “localhost” with the static IP address provided on google cloud

To start training models, use the train.py script in the root directory of the code. 

Ex: 
python train.py --batch_size=5 --model_name=fcn_crnn --dataset=full --epochs=10 --lr=.0001 --num_classes=4 --env_name=pretrained_true --least_cloudy=True --hidden_dims=32 --weight_decay=0 --patience=4 --weight_scale=.9 --pretrained=True
(is the actual training command we used)

Other models can be set using different model names (--model_name=bidir_clstm, for example), and additional hyperparameter tuning settings can be set by invoking the appropriate flags. 

See `get_train_parser` in `util.py` for full details. 

