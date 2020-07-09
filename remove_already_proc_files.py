import os

if __name__ == '__main__':
# Checks if files exist in hold_dir and if so
# removes them from proc_dir
#
# Is because a copy of file should be made after
# running perform_snl on it -- should avoid repeat runs
    snl_dir = './simulated_trinity_data/'
    hold_dir = './temp/'
    if os.path.isdir(hold_dir):
        existing_files = os.listdir(snl_dir)
        proc_files = [os.path.basename(x) for x in os.listdir(hold_dir)]
        for fpath in existing_files:
            exist_name = os.path.basename(fpath)
            if exist_name in proc_files:
                os.remove(snl_dir+exist_name)
        print("\nFiles in both \'%s\' and \'%s\' were removed from \'%s\'\n"%(snl_dir, hold_dir, snl_dir))
    else:
        print("\nOne of the specified directories does not exist\n")
