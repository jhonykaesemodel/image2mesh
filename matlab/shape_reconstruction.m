data_paths;
add_paths;

%% Parameters
file_params_gt = 'FFD_LC_IDX_params.h5';
file_idx_hat_train = 'model_classifier_idx_params_b1_e1000_train.mat';
file_idx_hat_test = 'model_classifier_idx_params_b1_e1000_test.mat';
file_params_hat_train = 'model_fnn_params_b1_e2000_train';
file_params_hat_test = 'model_fnn_params_b1_e2000_test';

is_plot_confusion = false;
idx_sample = 3; % example to be reconstructed if not using the loop to iterate over all test samples

% class = 'chair';
% class = 'sofa';
% class = 'bus';
class = 'aeroplane';
% class = 'bicycle';
% class = 'car';
% class = 'motorbike';
% class = 'diningtable';

% plots setup
colorModel = [38 139 210]/255;
edgeColor = [0  43  54]/255;
colorGCA = [1 1 1];
colorLine = [0  43  54]/255;
markSizeLatt = 8;
markSize3D = 10;
lineWidLatt = 0.5;
lightPos_x = 0;
lightPos_y = -1;
lightPos_z = 1;
light_style = 'infinite';

%% Define paths
classUID = class2uid(class);

data_dir = fullfile(data_dir, classUID);
graphfile = fullfile(graph_dir, [classUID, '.mat']);
dir_params_gt = fullfile(ffd_dir, classUID , 'rendered', 'parameters', file_params_gt);
dir_idx_hat_train = fullfile(data_dir , file_idx_hat_train);
dir_idx_hat_test = fullfile(data_dir , file_idx_hat_test);
dir_params_hat_train = fullfile(data_dir , file_params_hat_train);
dir_params_hat_test = fullfile(data_dir , file_params_hat_test);
images_dir = fullfile(ffd_dir, classUID, 'rendered', 'images');
obj_dir = fullfile(data_dir , 'obj_models');
if ~exist(obj_dir)
    mkdir(obj_dir);
end

%% Load graph
fprintf('Loading graph from %s...\n', graphfile);
fgraph = load(graphfile);
fgraph = fgraph.obj;

%% Load GT parameters
params_gt = h5read(dir_params_gt, '/GT');
idx_gt_train = round(params_gt(1:3500,127)*100);
idx_gt_test = round(params_gt(3501:end,127)*100);
params_gt_train = params_gt(1:3500,1:126);
params_gt_test = params_gt(3501:end,1:126);

load('indices_symmetry.mat')

%% Load estimated parameters
% indices
load(dir_idx_hat_train)
idx_hat_train = double(params_train);
load(dir_idx_hat_test)
idx_hat_test = double(params_test);
% parameters
load(dir_params_hat_train)
params_hat_train = squeeze(double(params_ffd_train));
load(dir_params_hat_test)
params_hat_test = squeeze(double(params_ffd_test));

%% Compute metrics
% metrics for the parameters estimation
mae_train = mae(params_gt_train, params_hat_train);
mse_train = mean(mean((params_gt_train - params_hat_train).^2));
rmse_train = sqrt(mean(mean((params_gt_train - params_hat_train).^2)));
fprintf("Parameters metrics (train set) - MSE: %f, RMSE: %f - MAE: %f \n", mse_train, rmse_train, mae_train);

mae_test = mae(params_gt_test, params_hat_test);
mse_test = mean(mean((params_gt_test - params_hat_test).^2));
rmse_test = sqrt(mean(mean((params_gt_test - params_hat_test).^2)));
fprintf("Parameters metrics (test set) - MSE: %f, RMSE: %f - MAE: %f \n", mse_test, rmse_test, mae_test);

% metrics for the indices classification
% convert to one-hot encoding first
idx_gt_train_oh = one_hot(idx_gt_train);
idx_gt_test_oh = one_hot(idx_gt_test);
idx_hat_train_oh = one_hot(idx_hat_train);
idx_hat_test_oh = one_hot(idx_hat_test);

% plot confusion matrix
if is_plot_confusion
    figure, plotconfusion(idx_gt_test_oh', idx_hat_test_oh');
    set(findobj(gca,'type','text'),'fontsize',5)
    
    figure, plotconfusion(idx_gt_train_oh', idx_hat_train_oh');
    set(findobj(gca,'type','text'),'fontsize',5)
end

% get confusion matrix
confusion_matrix = confusionmat(idx_gt_test, idx_hat_test);
accuracy = sum(idx_gt_test == idx_hat_test) / numel(idx_gt_test);
accuracyPercentage = 100*accuracy;

% get metrics
precision = @(y) diag(y)./sum(y,2);
recall = @(y) diag(y)./sum(y,1)';
f1scores = @(y) 2*(precision(y).*recall(y))./(precision(y)+recall(y));

precision_mean = mean(precision(confusion_matrix))*100;
recall_mean = nanmean(recall(confusion_matrix))*100;
f1scores_mean = nanmean(f1scores(confusion_matrix))*100;
fprintf("Indices metrics (test set) - Acc: %.2f, Precision: %.2f%%, Recall: %.2f%%, F1-Score: %.2f%% \n", ...
    accuracyPercentage, precision_mean, recall_mean, f1scores_mean);


is_experiment = true;
shape_cd = [];
shape_score = [];

for idx_sample = 3500:5000 % TODO: loop over all test samples 

% Get a 3D model from the graph
idx_gt_cat = [idx_gt_train; idx_gt_test];
idx_hat_cat = [idx_hat_train; idx_hat_test];
params_gt_cat = [params_gt_train; params_gt_test];
params_hat_cat = [params_hat_train; params_hat_test];
% using a GT index
model_ffd_gt = fgraph.nodes{idx_gt_cat(idx_sample)}.FFD;
params_gt = params_gt_cat(idx_sample,:);
% using an estimated index
model_ffd_hat = fgraph.nodes{idx_hat_cat(idx_sample)}.FFD;
params_hat = params_hat_cat(idx_sample,:);

% Recover the complete FFD parameters
% GT FFD parameters
vec_dP_gt = params_gt(1:96);
% apply symmetry constraint
vec_dP_full = zeros(192,1);
vec_dP_full(idx_values) = vec_dP_gt;
phi = symmetric_matrix(fgraph.nodes{1}.FFD.P, 3, 3, 3);
vec_dP_full = phi*vec_dP_full;
% apply FFD
deltaP = vec2mat(vec_dP_full ,3);
Phat = model_ffd_gt.P + deltaP;
Shat = model_ffd_gt.B * Phat;
model_ffd_gt_def = model_ffd_gt;
model_ffd_gt_def.vtx = Shat;
model_ffd_gt_def.Phat = Phat;

% estimated FFD parameters
vec_dP_hat = params_hat(1:96);
% apply symmetry constraint
vec_dP_full = zeros(192,1);
vec_dP_full(idx_values) = vec_dP_hat;
vec_dP_full = phi*vec_dP_full;
% apply FFD
deltaP = vec2mat(vec_dP_full ,3);
Phat = model_ffd_hat.P + deltaP;
Shat = model_ffd_hat.B * Phat;
model_ffd_hat_def = model_ffd_hat;
model_ffd_hat_def.vtx = Shat;
model_ffd_hat_def.Phat = Phat;

% Recover the linear combination (alphas) parameters
% get GT LC parameters
linearcomb_gt = params_gt(97:126);
% get estimated LC parameters
linearcomb_hat = abs(params_hat(97:126)) > 0.45; %TODO

ed = fgraph.edges(7,:);
[idd, vall] = find(ed == 1);

% apply linear combination
% take the edges for the linear combination
[~, idx_alphas] = find(linearcomb_hat == 1);
alphas = linearcomb_hat(idx_alphas);
alphas_full = linearcomb_hat;

if ~isempty(alphas)
    % perform the linear combination
    V = model_ffd_hat_def.vtx;
    for i = 1:numel(alphas)
        if idx_alphas(i) == idx_hat_cat(idx_sample)
            continue;
        end
        ffd_file = fullfile(shapenet_mat_dir, classUID, ...
            sprintf('node%02dto%02d.mat', idx_hat_cat(idx_sample), idx_alphas(i)));
        model = load(ffd_file, 'model');
        model = model.model;
        
        %         figure
        %         show_model(model, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        %             'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
        %         light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
        %         lighting flat
        %         set(gca,'Visible','off')
        %         grid off
        %
        %         set(gcf,'Units','Inches');
        %         pos = get(gcf,'Position');
        %         set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        %         filename = sprintf('node%02dto%02d_GT.png', idx_hat_cat(idx_sample), idx_alphas(i));
        %         print(gcf, '-dpng', '-r0', '-opengl', '-r600', filename)
        
        V = V + alphas(i)*model.vtx;
    end
    model_ffd_lc_hat_def = model_ffd_hat_def;
    model_ffd_lc_hat_def.vtx = V;
else
    model_ffd_lc_hat_def = model_ffd_hat_def;
end

% Shape metrics
% load the GT 3D model
render_name = sprintf('render%04d.obj', idx_sample);
model_aux = struct();
[model_aux.vertices, model_aux.faces] = read_obj(fullfile(root, 'FreeFormDeformation', classUID, 'rendered', render_name));
% get model index on the graph
for i = 1:numel(fgraph.nodes)
    if size(fgraph.nodes{i}.mesh, 1) == size(model_aux.faces, 1)
        model_gt_idx = i;
    end
end
model_gt = fgraph.nodes{model_gt_idx};
model_gt.vtx = model_aux.vertices;
model_gt.mesh = model_aux.faces;

% Chamfer distance
theta = 1e-3;
[CD, score] = distance_metrics(model_gt, model_ffd_lc_hat_def, theta);
%fprintf("CD: %.4f, Distance Score: %.4f \n", CD, score);

shape_cd = [shape_cd; CD];
shape_score = [shape_score; score];

% Save estimated model to OBJ file
% normalize model before saving to voxelize later
mean_vtx = mean(model_ffd_lc_hat_def.vtx', 2);
vtx_norm = bsxfun(@minus, model_ffd_lc_hat_def.vtx', mean_vtx);
model_ffd_lc_hat_def.vtx = vtx_norm';

if is_experiment
    fprintf("%s, CD: %.4f, DS: %.4f \n", render_name, CD, score);
end


% Show results using FFD + LC parameters
if is_experiment
    
    figure,
    subplot(2,3,1)
    show_model(model_ffd_gt, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    hold on
    show_FFD_lattice(model_ffd_gt.P, 3, 3, 3,  'MarkerSize', markSizeLatt, 'LineWidth', ...
        lineWidLatt, 'MarkerEdgeColor', edgeColor, 'ColorLine', colorLine);
    title('Model selected from the graph (GT)')
    
    subplot(2,3,2)
    show_model(model_ffd_gt_def, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    hold on
    show_FFD_lattice(model_ffd_gt_def.Phat, 3, 3, 3,  'MarkerSize', markSizeLatt, 'LineWidth', ...
        lineWidLatt, 'MarkerEdgeColor', edgeColor, 'ColorLine', colorLine);
    title('Deformed model with FFD (GT)')
    
    subplot(2,3,3)
    show_model(model_gt, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    title('Deformed model with FFD + Linear Combination (GT)')
    
    subplot(2,3,4)
    show_model(model_ffd_hat, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    hold on
    show_FFD_lattice(model_ffd_hat.P, 3, 3, 3,  'MarkerSize', markSizeLatt, 'LineWidth', ...
        lineWidLatt, 'MarkerEdgeColor', edgeColor, 'ColorLine', colorLine);
    title('Model selected from the graph (Hat)')
    
    subplot(2,3,5)
    show_model(model_ffd_hat_def, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    hold on
    show_FFD_lattice(model_ffd_hat_def.Phat, 3, 3, 3,  'MarkerSize', markSizeLatt, 'LineWidth', ...
        lineWidLatt, 'MarkerEdgeColor', edgeColor, 'ColorLine', colorLine);
    title('Deformed model with FFD (Hat)')
    
    subplot(2,3,6)
    show_model(model_ffd_lc_hat_def, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    title('Deformed model with FFD + Linear Combination (Hat)')
    
    %% Plots with GT image
    figure,
    subplot(1,3,1)
    image_name = sprintf('img%05d.png', idx_sample);
    filename = fullfile(images_dir, image_name);
    imshow(imread(filename));
    title('Input Image')
    
    subplot(1,3,2)
    show_model(model_gt, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    title('Deformed model with FFD + Linear Combination (GT)')
    
    subplot(1,3,3)
    show_model(model_ffd_lc_hat_def, 'FaceColor', colorModel, 'ColorGCA', colorGCA, 'isAnchor', false, ...
        'isLattice', false, 'isAxisLabel', false, 'isLattice', false, 'lighting', false);
    light('Position', [lightPos_x, lightPos_y, lightPos_z], 'Style', light_style);
    lighting flat
    title('Deformed model with FFD + Linear Combination (Hat)')
    
end

end % end loop for the iteration over samples

filename_shape_metrics = fullfile(data_dir,'shape_metrics_test.mat');
save(filename_shape_metrics, 'shape_cd', 'shape_score')
