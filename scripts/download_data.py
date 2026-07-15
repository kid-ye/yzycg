import wfdb
import os
import argparse
from tqdm import tqdm

def download_ptbdb_subset(num_patients=None, data_dir='data/ptbdb'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print('Fetching record list from PhysioNet...')
    records = wfdb.get_record_list('ptbdb')
    
    patient_ids = sorted(list(set([r.split('/')[0] for r in records])))

    if num_patients:
        target_patients = patient_ids[:num_patients]
        print(f'Downloading subset of PTBDB ({len(target_patients)} patients) to {data_dir}...')
    else:
        target_patients = patient_ids
        print(f'Downloading FULL PTBDB dataset ({len(target_patients)} patients) to {data_dir}...')
    
    for pid in tqdm(target_patients, desc='Downloading Patients'):
        patient_records = [r for r in records if r.startswith(pid)]
        for rec in patient_records:
            # Check if files already exist to support resuming
            dat_path = os.path.join(data_dir, rec + '.dat')
            hea_path = os.path.join(data_dir, rec + '.hea')
            
            if os.path.exists(dat_path) and os.path.exists(hea_path):
                continue
                
            try:
                wfdb.dl_files('ptbdb', data_dir, [rec + '.dat', rec + '.hea'])
            except Exception as e:
                tqdm.write(f'Failed to download {rec}: {e}')

    print(f'Download complete! {len(target_patients)} patients processed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download PTBDB dataset')
    parser.add_argument('--count', type=int, default=None, help='Number of patients to download (default: All)')
    args = parser.parse_args()
    
    download_ptbdb_subset(num_patients=args.count)