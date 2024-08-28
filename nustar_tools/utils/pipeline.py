import os
import shutil
import datetime

from ..utils import utilities, solar_convert


NU_DATA_DIR = utilities.get_cwd() + 'data/'
TMP_DIR = NU_DATA_DIR + 'tar_temp/'
TAR_DIR = utilities.get_cwd() + 'tars/' # The input tar directory
TAR_ARCHIVE = NU_DATA_DIR + 'tar_archive/' # The directory for tars that have been untarred.
ESSENTIAL_DIRS = [NU_DATA_DIR, TAR_ARCHIVE, TAR_DIR]
EVENT_DATA_EXTENSIONS = ['A06_cl.evt', 'B06_cl.evt'] # File extensions of the event data
TAR_FORMAT = '{date:8}{anything}.tar'


def move_and_untar(tar_file):
    """
    Inputs:
        file [string]: name of the .tar file to be extracted.
    
    Outputs:
        id_dir_list [list]: List of IDs extracted from the .tar file
    """
    
    # Check that the tar file has a properly formatted name.
    try:
        p = utilities.parse.parse(TAR_FORMAT, tar_file)
        date = p['date']
    except:
        print('Tar file name format is incorrect')

    src = TAR_DIR + tar_file
    dest = TAR_ARCHIVE + tar_file
    
    # Move the tar file to the tar archive for safety reasons.
    # If the destination already exists, append the current timestamp.
    if os.path.isfile(TAR_ARCHIVE + dest[utilities.find_nth(dest, '/', dest.count('/'))+1:]):
        new_name = dest.split('.tar')[0] + '_' + datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S') + '.tar'
        dest = new_name
    shutil.move(src, dest)
            
    date_dir = NU_DATA_DIR + date + '/'
    utilities.create_directories(date_dir)

    pre_list = os.listdir(date_dir)
    os.system(f'tar -zxvf {dest} -C {date_dir}')
    id_list = os.listdir(date_dir)
    
    id_dirs = [] # List of IDs extracted from the tar file
    for data_id in id_list:
        if data_id not in pre_list:
            id_dir = date_dir + data_id + '/'
            id_dirs.append(id_dir)
    
    return id_dirs


def run_pipeline(id_dir):

    cmd = './run_pipe_solar.sh ' + id_dir
    os.system('heainit')
    os.system(cmd)


def process_tar(tar_file):
    """
    Executes the following instructions for handling new data sets:
        - Moves the .tar file from Downloads to the NuSTAR data folder.
        - Untars the .tar file into the appropriate folder labelled by observation ID.
        - Runs the pipeline on the extracted data.
        - Generates the relevant solar coordinate data for maps.
        - Automatically plots the maps
        - Automatically plots the livetime data.
    """
    
    print('Processing ' + tar_file)
    id_dirs = move_and_untar(tar_file)
    print(f'Found the following IDs: {id_dirs}')

    for id_dir in id_dirs:
        print(f'Running pipeline on {id_dir}.')
        run_pipeline(id_dir)

        print('Generating solar data file.')
        solar_convert.generate_solar_data(id_dir)


def auto_pipeline():
    """
    Checks the /tars/ folder for new data downloads and automatically handles them.
    """
    
    # Check that the essential directories exist. Create them if they do not.
    utilities.create_directories(ESSENTIAL_DIRS)

    print(f'Checking {TAR_DIR} for new data sets.')
    files = os.listdir(TAR_DIR)
    tar_files = []
    for tar_file in files:
        if tar_file.endswith('.tar'):
            tar_files.append(tar_file)
    
    print(f'Found tar files {tar_files}')
    if tar_files:
        for tar_file in tar_files:
            process_tar(tar_file)
    else:
        print('No new data sets.')