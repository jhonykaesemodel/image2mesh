data_paths;
add_paths;

% class = 'chair';
% class = 'sofa';
% class = 'bus';
class = 'aeroplane';
% class = 'bicycle';
% class = 'car';
% class = 'motorbike';
% class = 'diningtable';

%% Define paths
classUID = class2uid(class);
data_dir = fullfile(data_dir, classUID);
dataset_dir = root;
obj_dir = fullfile(data_dir, 'obj_models');

%% Get metrics
filename = fullfile(data_dir, 'shape_metrics_test.mat');
load(filename) 
CD = nanmean(shape_cd); % TODO
DIST = mean(shape_score);

%% Get IoU
filename = fullfile(obj_dir, 'all_iou_test.mat');
load(filename);
IOU = mean(all_iou_test);

fprintf("Dist: %.3f, CD: %.3f, IOU: %.3f \n", DIST, CD, IOU);
