%% 1. Download datasets and install dependencies
% Download the ShapeNetCore.v1 and PASCAL3D+_release1.1 datasets; 
% Place them on e.g. C:\datasets\ShapeNetCore.v1 and C:\datasets\PASCAL3D+_release1.1.
% Download the ShapeNetAnchors data from % https://github.com/jhonykaesemodel/compact_3D_reconstruction;
% The directory contains ~30 3D models IDs (folder naming) from ShapeNetCore + 3D anchors for the
% 8 classes used in the paper. The IDs and 3D anchors are necessary to build the embedding graphs.
% Install CVX, Python 3, PyTorch and all modules needed according to the Python files.

%% 2. Clone the Compact 3D Representation project
% Create dev directory
dev_dir = 'C:\dev\'; % Change the path if needed
if ~exist(dev_dir, 'dir')
    mkdir(dev_dir)
end
cd(dev_dir);
command = 'git clone https://github.com/jhonykaesemodel/compact_3D_reconstruction.git';
system(command, '-echo');

%% 3. Create the embedding graph G - E.g. for the aeroplane class
cd(strcat(dev_dir, 'compact_3D_reconstruction'));
add_paths;
data_paths; % Change the paths accordingly.
% It takes ~10h in a i7-56000U CPU with 16 GB of RAM. The data needs ~12 GB of disk space.
% The last file to be computed will be node30to29.mat (aeroplane class).
% Change the path of line 39 in the python\vxl.py file accordingly. E.g.
% filedir = os.path.join('C:\\datasets\\ShapeNetMat.v1\\', clsuid).
exp_create_graph; % This will create the graph for the aeroplane class

%% 4. Run the 3D reconstruction from a single image on the Pascal3D+ dataset
% This is necessary because we used the estimated linear combination parameters 
% to fit a normal distribution to generate new data later
exp_full_pascal3D; % It will run on the aeroplane class

%% 5. Generate data using the embedding graph
cd(fullfile(dev_dir, 'image2mesh', 'matlab'));
add_paths; % Change the paths accordingly
data_paths; % Change the paths accordingly
% It will take a while to compute...
generate_data; % generate data for the aeroplane class

%% 6. Train/Test the learning framework using the generated data
% Make sure the paths of the Python files below are consistent (''' Define paths ''')
python_dir = fullfile(dev_dir, 'image2mesh', 'python');
% Convolutional autoencoder to extract the image’s latent space
system(['python ' python_dir '\convolutional_autoencoder.py -c ''aeroplane'''], '-echo');
% Multi-label classifier to classify the image’s latent space to a graph node index
system(['python ' python_dir '\multilabel_classifier.py -c ''aeroplane'''], '-echo');
% Feedforward network to regress the image’s latent space to a compact shape parametrization (FFD + linear combination)
system(['python ' python_dir '\feedforward_network.py -c ''aeroplane'''], '-echo');

%% 7. 3D reconstruction from a single image & Evaluation
shape_reconstruction; % Reconstructs one example from the aeroplane test set
% get voxel models from all obj files to compute the IoU
current_dir = pwd;
cd(fullfile(python_dir, 'voxel'));
system('python -c from get_voxels import prepare_data; prepare_data("aeroplane")', '-echo');
system('python -c from get_all_iou import compute_iou; compute_iou("aeroplane")', '-echo');
cd(current_dir);
% evaluate reconstruction
eval_shape_metrics;
