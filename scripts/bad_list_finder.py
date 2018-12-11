if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str,
                        help='home dir',
                        default='/home/data')
    parser.add_argument('--country', type=str,
                        help='which country',
                        default='ghana')                    
    args = parser.parse_args()
    home = args.home
    country = args.country
    
    bad_list_s1 = list()
    bad_list_s2 = list()
    
    s1_dir = os.path.join(home, country, 's1_npy')
    s1_fnames = [f for f in os.listdir(s1_dir) if f.endswith('.npy')]
    s1_fnames = [f for f in s1_fnames if 'cloudmask' not in f]
    s1_ids = [f.split('_')[-1].replace('.npy', '') for f in s1_fnames]

    for i in range(len(s1_fnames)):
        temp = np.load(os.path.join(s1_dir,s1_fnames[i]))
        if temp.shape[3]<MIN_TIMESTAMPS:
            bad_list_s1.append(s1_ids[i])
            
    s2_dir = os.path.join(home, country, 's2_npy')
    s2_fnames = [f for f in os.listdir(s2_dir) if f.endswith('.npy')]
    s2_fnames = [f for f in s2_fnames if 'cloudmask' not in f]
    s2_ids = [f.split('_')[-1].replace('.npy', '') for f in s2_fnames]

    for i in range(len(s2_fnames)):
        temp = np.load(os.path.join(s2_dir,s2_fnames[i]))
        if temp.shape[3]<MIN_TIMESTAMPS:
            bad_list_s2.append(s2_ids[i])
            
    bad_list_s1.append(bad_list_s2)
    bad_list = . np.unique(np.array(bad_list_s1))
    np.save(os.path.join(home, country, 'bad_timestamp_grids_list.npy'), bad_list)
    
