""" 
Converts planet imagery for grid IDs from Radiance to TOA Reflectance based on coefficients 
provided in xml files associated with every image and stacks them in time into npy files.

Based on this tutorial: 
https://developers.planet.com/tutorials/convert-planetscope-imagery-from-radiance-to-reflectance/
"""

import rasterio
import numpy as np
import argparse
import json
import os
import sys

from xml.dom import minidom
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sys.path.insert(0, '../')
import mk_data_cube 

import pdb

def get_radiance(filename):
    ''' 
    2.) Extract data from each spectral band:
        Load red and NIR bands - note all PlanetScope 
        4-band images have band order BGRN
    '''
    rad = {}
    with rasterio.open(filename) as src:
        rad['blue'] = src.read(1)
    with rasterio.open(filename) as src:
        rad['green'] = src.read(2)
    with rasterio.open(filename) as src:
        rad['red'] = src.read(3)
    with rasterio.open(filename) as src:
        rad['nir'] = src.read(4)
    return rad

def extract_coeffs(xml_fname):
    '''
    3.) Extract the coefficients from xml file
    '''
    xmldoc = minidom.parse(xml_fname)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)

    print("Conversion coefficients:", coeffs)
    return coeffs

def rad2toa(img_fname, xml_fname):
    '''
    Convert Radiance to Reflectance using coefficients
    '''

    rad = get_radiance(img_fname)
    coeffs = extract_coeffs(xml_fname)
    
    if any(coeffs):
        # Multiply the Digital Number (DN) values in each band by the TOA reflectance coefficients
        scale = 10000.
        blue_toa = np.expand_dims(scale * (rad['blue'] * coeffs[1]), axis=0)
        green_toa = np.expand_dims(scale * (rad['green'] * coeffs[2]), axis=0)
        red_toa = np.expand_dims(scale * (rad['red'] * coeffs[3]), axis=0)
        nir_toa = np.expand_dims(scale * (rad['nir'] * coeffs[4]), axis=0)

        toa = np.concatenate([blue_toa, green_toa, red_toa, nir_toa], axis=0)
        print("Red band radiance is from {} to {}".format(np.amin(rad['red']), np.amax(rad['red'])))
        print("Red band reflectance is from {} to {}".format(np.amin(red_toa), np.amax(red_toa)))
        return toa
    else:
        return None

def main(args):
    #pdb.set_trace()
    # Get lists of imagery and xml files
    cur_path = os.path.join(args.home, args.country, args.source)
    files = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.tif')]
    xmls = [os.path.join(cur_path, f) for f in os.listdir(cur_path) if f.endswith('.xml')]

    grid_numbers = [f.split('/')[-1].split('_')[0] for f in files]
    grid_numbers.sort()

    for grid_idx, grid in enumerate(sorted(set(grid_numbers))):
        print('cur grid: ', grid)
        print('cur idx: ', grid_idx)

        cur_grid_files = [f for f in files if f.split('/')[-1].startswith(grid + '_')]
        cur_grid_files.sort() # sorts in time

        cur_grid_xmls = [f for f in xmls if f.split('/')[-1].startswith(grid + '_')]
        cur_grid_xmls.sort() # sorts in time

        # Image files and xml files should match up
        assert len(cur_grid_files) == len(cur_grid_xmls)

        dates = []
        data_array = []
        npix = []
        for idx, fname in enumerate(cur_grid_files):
            print('img: ', fname)
            print('xml: ', cur_grid_xmls[idx])
 
            statinfo = os.stat(fname)
            # Check if tif file is empty
            if statinfo.st_size == 0:
                logfile=open("logfile_errors.txt","a+")
                logfile.write("Image is empty, passed: " + fname + "\n")
                logfile.close()
                pass
            else:
                with rasterio.open(fname) as src:
                    cur_img = src.read()
                    npix.append(cur_img.shape[1])
                    # convert to toa reflectance
                    toa_img = rad2toa(fname, cur_grid_xmls[idx])
                    # Check if toa image exists (DNE if coefficients DNE)
                    if toa_img is None:
                        logfile=open("logfile_errors.txt","a+")
                        logfile.write("TOA img is none, passed: " + fname + "\n")
                        logfile.close()
                        pass
                    else:
                        # Desired behavior given no errors. Append data and dates to lists.
                        data_array.append(np.expand_dims(toa_img, axis=3))
                        tmp = fname.split('/')[-1].split('_')[2]
                        tmp = '-'.join([tmp[:4], tmp[4:6], tmp[6:]])
                        dates.append(tmp)

        # Get fname for output files
        tmp = fname.split('/')
        tmp[-2] = tmp[-2] + '_npy'
        tmp[-1] = '_'.join([args.source, args.country, tmp[-1].split('_')[0] + '_toa'])
        output_fname = "/".join(tmp)
        print('out: ', output_fname)

        if not os.path.exists('/'.join(output_fname.split('/')[:-1])):
            os.makedirs('/'.join(output_fname.split('/')[:-1]))

        # store and save metadata
        meta = {}
        meta['dates'] = dates
        with open(output_fname + '.json', 'w') as fp:
            json.dump(meta, fp)

        # save image stack as .npy
        # Check that all images in list have the same height / width
        if len(np.unique(npix)) == 1:
            data_array = np.concatenate(data_array, axis=3) 
            data_array = data_array.astype(np.uint16)
            print('min in dataarray: ', np.min(data_array))
            print('max in dataarray: ', np.max(data_array))
            np.save(output_fname + '.npy', data_array)
        else:
            cur_npix = np.max(np.unique(npix))
            data_tmp = np.zeros((args.bands, cur_npix, cur_npix, len(data_array)))
            for arr_idx, arr in enumerate(data_array):
                data_tmp[:, :arr.shape[1], :arr.shape[2], arr_idx] = np.squeeze(arr)
            print('min in dataarray: ', np.min(data_tmp))
            print('max in dataarray: ', np.max(data_tmp))
            data_tmp = data_tmp.astype(np.uint16)
            np.save(output_fname + '.npy', data_tmp)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str,
                        help='Path to directory containing data.',
                        default='/home/data')
    parser.add_argument('--country', type=str,
                        help='Country of interest: "ghana", "southsudan", "tanzania"',
                        default='ghana')
    parser.add_argument('--source', type=str,
                        help='Satellite source.',
                        default='planet')
    parser.add_argument('--bands', type=int,
                        help='Number of image bands.',
                        default=4)
    args = parser.parse_args()
    main(args)


