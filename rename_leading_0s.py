import os
import argparse


def rename(old_fname, tif_content, num_digits, dry_run):
    """
    This function assumes that old_fname of tif_content 
      'mask' ends with path/COUNTRY_DIMxDIM_###.tif 
      'data' ends with path/SOURCE_COUNTRY_ORBIT_###_YYYY-MM-DD.tif

    where ### corresponds to a grid ID, and can vary from # to ##### 
    """
    fname_split = old_fname.split('_')
    
    if tif_content == 'mask':
        assert '.tif' in fname_split[-1]
        fname_split[-1] = fname_split[-1].replace('.tif', '').zfill(num_digits) + '.tif'
    elif tif_content == 'data':
        assert fname_split[-2].isdigit()
        fname_split[-2] = fname_split[-2].zfill(num_digits)

    new_fname = '_'.join(fname_split)
    if dry_run:
        print 'Renaming {} to {}'.format(old_fname, new_fname)
    else:
        print 'Not a dry run!'
        print 'Renaming {} to {}'.format(old_fname, new_fname)
        os.rename(old_fname, new_fname)


def get_fnames(directory):
    """ 
    Returns a list of *.tif filenames given a directory
    """
    print 'directory: ', directory
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]


def main(directory, tif_content, num_digits, dry_run):
    fnames = get_fnames(directory)
    for f in fnames:
        rename(f, tif_content, num_digits, dry_run)    


if __name__ == '__main__':
    """
    Renames grid IDs with leading 0's so that sorting of filenames is in ascending order

    To call on masks,
       python rename_leading_0s.py --dir /home/roserustowicz/data/COUNTRY/raster --tif_content mask

    To call on data, 
       python rename_leading_0s.py --dir /home/roserustowicz/data/COUNTRY/SOURCE --tif_content data 
  
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=str, default='/home/roserustowicz/data/Ghana/raster_64x64',
        help='Directory in which to replace grid IDs with leading 0''s')
    parser.add_argument('--dry_run', dest='dry_run', action='store_true')
    parser.add_argument('--num_digits', type=int, default=5,
        help='Number of digits that ID should contain (default: 5). Given default of "5", ID "34" --> "00034"')
    parser.add_argument('--tif_content', type=str, default='mask', 
        help='Defines what is contained in the tif (options: "mask", "data")')
    args = parser.parse_args()

    main(args.dir, args.tif_content, args.num_digits, args.dry_run)

