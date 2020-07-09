import os

if __name__ == '__main__':
# Checks if files exist in hold_dir and if so
# removes them from proc_dir
#
# Is because a copy of file should be made after
# running perform_snl on it -- should avoid repeat runs
    snl_dir = './simulated_trinity_data/'
    hold_dir = './temp/'

    existing_files = os.listdir(snl_dir)
    proc_files = [os.path.basename(x) for x in os.listdir(hold_dir)]
    for fpath in existing_files:
        exist_name = os.path.basename(fpath)
        if exist_name in proc_files:
            print(exist_name)
            os.remove(snl_dir+exist_name)
