This readme explains how data is downloaded and formatted to be used in our models. 

1.) Download the data from a google bucket. 

    ex:  `sudo gsutil -m cp -r gs://es262-croptype/Tanzania/* /home/data/tanzania`

    This copies files from the bucket called `es262-croptype/Tanzania` into a local
    folder called `/home/data/tanzania`. The folder should contain directories 
    `raster`, `s1`, and `s2`. 

    `raster` - contains labels of the grid IDs. Values in these labels correspond
               to field IDs, which are referred to as `geom_ID`s in the csv files
               for each country. 
    `s1` - contains Sentinel-1 images for gridIDs
    `s2` - contains Sentinel-2 images for gridIDs

2.) Next, we want to rename the data such that the gridIDs in the filenames have 
    leading zeros. We will use the function `rename_w_leading_0s.py`, located in the
    `scripts` folder of the `crop-type-mapping` repository. To double check that the
    function performs what you want it to, you can use the --dry_run flag
    
    To call on labels,
       python rename_w_leading_0s.py --dir /home/data/COUNTRY/raster --tif_content mask
 
       For our working example, we can run:
       `python rename_w_leading_0s.py --dir /home/data/tanzania/raster --tif_content mask`


    To call on data, 
       python rename_leading_0s.py --dir /home/data/COUNTRY/SOURCE --tif_content data --country COUNTRY
  
       For our working example, we can run:
       `python rename_w_leading_0s.py --dir /home/data/tanzania/s1 --tif_content data --country tanzania`
       `python rename_w_leading_0s.py --dir /home/data/tanzania/s2 --tif_content data --country tanzania`

3.) There may be some cases where data exists in the s1 and s2 folders, while their corresponding
    label in the raster folder doesn't actually contain any labels. To check for these cases, run:

    `python remove_invalid_grids.py`

    If you open the script, you can specify parameters at the bottom. Be sure to run with `dryrun = 1`
    before actually deleting anything, just to be sure the script acts as expected.

4.) We want to create data stacks for each grid, so that we can have one cube per grid that
    incorporates all of our temporal data for that grid. We will save these files as .npy files so 
    that they can be easily read in. In saving these files, we also want to save a corresponding 
    .json file that will keep track of the dates of images listed in the array. 

    To run: `python mk_data_cube.py`, located in the `crop-type-mapping/scripts` directory. 

   If you open the script, you will specify parameters at the bottom! 

5.) We also want to generate cloud masks for each of the Sentinel-2 npy files defined in part 4.  
    To do so, edit the parameters at the bottom of the file and run:

    `python crop-type-mapping/scripts/cloud_classifier.py`

6.) Finally, we want to change the values in the raster files so that the labels correspond to crop type
    rather than field ID. To do so, run:
