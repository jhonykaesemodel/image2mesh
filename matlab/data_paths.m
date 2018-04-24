% Change these paths accordingly
root = 'C:\datasets';
data_dir = 'C:\data\img2mesh'; % where all data will be saved

% Pascal 3D+
PASCAL3D_dir = fullfile(root, 'PASCAL3D+_release1.1');

% Results
ffd_dir = fullfile(root, 'FreeFormDeformation');
pascal_result_dir = fullfile(root, 'PascalResult');
anchor_dir = fullfile(root, 'ShapeNetAnchors');
graph_dir = fullfile(root, 'ShapeNetGraph');
shapenet_mat_dir = fullfile(root, 'ShapeNetMat.v1');
pascal_images_dir = fullfile(root, 'PascalImages');
