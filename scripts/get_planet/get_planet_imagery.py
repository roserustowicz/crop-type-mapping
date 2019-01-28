import os
import sys
import numpy as np
import argparse
import rasterio
import rasterio.features
import rasterio.warp

from os import environ, makedirs, path 
from osgeo import gdal, gdal_array, ogr
from pathlib import Path
from geojson import Polygon, Feature, FeatureCollection, dump
from requests import Session, get, post
from requests.auth import HTTPBasicAuth
from retrying import retry
from sys import stdout

from functools import partial
from itertools import repeat
from multiprocessing import Pool # , freeze_support
import pdb

sys.path.insert(0, '../../')

from util import str2bool
from constants import * 

def create_tif_mask(in_fname, out_fname):
    # Open the raster file
    raster = gdal.Open(in_fname, gdal.GA_ReadOnly)
    band_1 = raster.GetRasterBand(1)

    # Read the data as a numpy array
    data_1 = gdal_array.BandReadAsArray(band_1)
    
    # Create a boolean band with all 1's 
    mask = (data_1 >= 0).astype(int)
    assert np.mean(mask) == 1

    # Write the output mask
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(out_fname, raster.RasterXSize, raster.RasterYSize, 1, band_1.DataType)
    gdal_array.CopyDatasetInfo(raster, ds_out)
    band_out = ds_out.GetRasterBand(1)
    gdal_array.BandWriteArray(band_out, mask)

    # Close the datasets
    band_1 = None
    raster = None
    band_out = None
    ds_out = None


def create_shp_from_mask(shp_fname, mask_fname):
    # Create a shapefile from the mask that was just created 
    # Initialize the information for the shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shape_file = driver.CreateDataSource(shp_fname + ".shp")
    layer = shape_file.CreateLayer(shp_fname, srs=None)

    # Open the mask and polygonize it into the specified shape file
    mask = gdal.Open(mask_fname)
    mask_band = mask.GetRasterBand(1)
    gdal.Polygonize(mask_band, mask_band, layer, -1, [], callback=None)


def get_geojson_from_img(in_fname, out_fname):
    # From https://rasterio.readthedocs.io/en/latest/
    # Create a geojson file from an input filename
    with rasterio.open(in_fname) as dataset:
        # Read the dataset's valid data mask as a ndarray
        mask = dataset.dataset_mask()
        # Extract feature shapes and values from the array
        count = 0
        for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
            coordinates = geom['coordinates']
            geojson_geom = {"type": "Polygon", "coordinates": coordinates} 
            assert count == 0
            count += 1

        polygon = Polygon(coordinates)
        features = []
        features.append(Feature(geometry=polygon, properties={"": ""}))
        feature_collection = FeatureCollection(features)
         
        with open(out_fname, 'w') as f:
            dump(feature_collection, f)
        
        return geojson_geom


# "Wait 2^x * 1000 milliseconds between each retry, up to 10
# seconds, then 10 seconds afterwards"
@retry(wait_exponential_multiplier=1000,wait_exponential_max=10000)
def activate_something(session, item_type, item_id, asset_type):
    item = session.get(("https://api.planet.com/data/v1/item-types/" + "{}/items/{}/assets/").format(item_type, item_id))

    # raise an exception to trigger the retry
    if item.status_code == 429:
        raise Exception("rate limit error")
    if item.status_code == 202:
        raise Exception("activation request still processing")

    item_activation_url = item.json()[asset_type]["_links"]["activate"]
    response = session.post(item_activation_url)
    #print('response status code {}: {}'.format(asset_type, response.status_code))
                        
    if response.status_code == 429:
        raise Exception("rate limit error")
    if item.status_code == 202:
        raise Exception("activation request still processing")

                        
@retry(wait_exponential_multiplier=1000,wait_exponential_max=10000)
def download_something(session, item_type, item_id, asset_type, save_dir, field_id, ext):
    geojson_fname = 'tmp_geojson_'+field_id+'.geojson'

    item = session.get(("https://api.planet.com/data/v1/item-types/" + "{}/items/{}/assets/").format(item_type, item_id))

    if item.status_code == 429:
        raise Exception("rate limit error")    
    if item.status_code == 202:
        raise Exception("download request still processing")

    item_download_url = item.json()[asset_type]["location"]
    fname = save_dir + field_id + '_' + item_type + '_' + item_id + '_' + asset_type + ext

    # check if already downloaded first
    my_file = Path(fname)
    if my_file.is_file():
        print("{} file is already downloaded.".format(asset_type))
    else:
        with open(fname, "wb") as f:
            print("Downloading {}".format(fname))
                             
            if 'tif' in ext:
                # Download the subregion of interest
                vsicurl_url = '/vsicurl/' + item_download_url
                output_file = fname
                gdal.Warp(output_file, vsicurl_url, dstSRS='EPSG:4326', 
                          cutlineDSName=geojson_fname, cropToCutline=True) 
            elif 'xml' in ext:
                r = get(item_download_url, stream=True, allow_redirects=True)
                total_length = r.headers.get('content-length')
                if total_length is None:  # no content length header
                    f.write(r.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for dataa in r.iter_content(chunk_size=4096):
                        dl += len(dataa)
                        f.write(dataa)
                        done = int(50 * dl / total_length)
                        stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                        stdout.flush()

def main(raster_dir, save_dir, activate, download, item_type):

    # filter images acquired in a certain date range
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": { "gte": "2017-05-01T00:00:00.000Z", 
                    "lte": "2017-10-31T23:59:59.999Z" }}

    # filter images with <= 10% cloud coverage
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": { "lte": 0.10 }}

    # Read through raster images to get AOIs, then use these AOIs to clip planet imagery?
    raster_fnames = [path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]
    print('total raster fnames: ', len(raster_fnames))
    raster_fnames.sort()

    for idx1 in range(len(raster_fnames)):
        print('\n\nRaster file {} of {}'.format(idx1, len(raster_fnames)))
        fname = raster_fnames[idx1]
        field_id = fname.split('_')[-1].replace('.tif', '')

        # If the raster is all zeros, skip it
        gtif = gdal.Open(fname, gdal.GA_ReadOnly)
        bnd = gtif.GetRasterBand(1)
        bnd_arr = gdal_array.BandReadAsArray(bnd)

        if np.sum(bnd_arr) == 0:
            print('\n\nPassed on {} of {}, fname {}, raster is all zeros'.format(idx1, len(raster_fnames), fname))
        else:
            if download:
                create_tif_mask(fname, out_fname='tmp_mask_'+field_id+'.tif')
                create_shp_from_mask(shp_fname='tmp_shp_'+field_id, mask_fname='tmp_mask_'+field_id+'.tif')    
   
            # Define a geometry filter
            geojson_geom = get_geojson_from_img(fname, 'tmp_geojson_'+field_id+'.geojson')
            geometry_filter = {"type": "GeometryFilter", "field_name": "geometry", "config": geojson_geom}
 
            # Define overall AndFilter that combines our geo and date filters
            current_filter = {"type": "AndFilter", "config": [geometry_filter, date_range_filter, cloud_cover_filter]}
        
            # Search API request object
            search_endpoint_request = {"item_types": [item_type], "filter": current_filter}
            result = post(
                'https://api.planet.com/data/v1/quick-search',
                auth=HTTPBasicAuth(environ['PL_API_KEY'], ''),
                json=search_endpoint_request)

            data = result.json()

            try:
                to_unicode = unicode
            except NameError:
                to_unicode = str

            print('Num to activate: ', len(data['features']))       

            # Setup authentication
            session = Session()
            session.auth = (environ['PL_API_KEY'], '')
       
            item_ids = []
            for idx in range(len(data['features'])):
                item_ids.append(data['features'][idx]['id'])
            item_ids.sort()
             
            if activate:
                # Make request for analytic asset
                for asset_type in ["analytic", "analytic_xml"]:
                     with Pool(8) as pool1:
                         pool1.starmap(activate_something, zip(repeat(session), repeat(item_type), item_ids, repeat(asset_type)))
            if download:
                if not path.exists(save_dir):
                    makedirs(save_dir)

                # Download the assets
                for asset_type in ["analytic"]:
                    with Pool(8) as pool2:
                        pool2.starmap(download_something, zip(repeat(session), repeat(item_type), item_ids, repeat(asset_type), repeat(save_dir), repeat(field_id), repeat(".tif")))
                for asset_type in ["analytic_xml"]:
                    with Pool(8) as pool3:
                        pool3.starmap(download_something, zip(repeat(session), repeat(item_type), item_ids, repeat(asset_type), repeat(save_dir), repeat(field_id), repeat(".xml")))
                        
                print('Downloading Started ... ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raster_dir', type=str, help='Path to directory of raster data',
                        default=GHANA_RASTER_DIR)
                        #default=LOCAL_DATA_DIR + '/ghana/raster/')
    parser.add_argument('--save_dir', type=str, 
                        help='Path to save planet assets to',
                        default=GCP_DATA_DIR + '/ghana/planet/')
                        #default=LOCAL_DATA_DIR + '/ghana/planet')
    parser.add_argument('--activate', type=str2bool,
                        help="Activate planet items",
                        default=False)
    parser.add_argument('--download', type=str2bool,
                        help="Download planet items",
                        default=False)
    parser.add_argument('--item_type', type=str,
                        help="Planet item type to download/activate",
                        default="PSScene4Band")

    args = parser.parse_args()
    main(args.raster_dir, args.save_dir, args.activate, args.download, args.item_type)
