add_paths;
data_paths;

% classes = { 'chair', ...
%             'sofa', ...
%             'bus', ...
%             'aeroplane', ...
%             'bicycle', ...
%             'car', ...
%             'motorbike', ...
%             'diningtable'};

classes = {'aeroplane'};

num_cores = 2;
parpool(num_cores);
for i = 1:numel(classes)
    
    c = classes{i};
    classUID = class2uid(c);
    
    GM = GenerativeModel(classUID);
    
    %% Generate random FFD model
    % N = 1;
    % rdnModel = GM.randomModel(N);
    % model_FFD = GM.modelFFD{N};
    %
    % % visualize
    % colorGCA = [7 54 66]/255;
    % colorModel = [38 139 210]/255;
    %
    % figure,
    % subplot(1,3,1)
    % show_model(model_FFD, 'FaceColor', colorModel, 'ColorGCA', colorGCA, ...
    %         'MarkerSize', 20, 'isAnchor', false, 'isLattice', true);
    % title('Original FFD')
    %
    % subplot(1,3,2)
    % show_model(rdnModel.model_FFD_def, 'FaceColor', colorModel, 'ColorGCA', colorGCA, ...
    %         'MarkerSize', 20, 'isAnchor', false, 'isLattice', true);
    % title('Deformed lattice')
    %
    % subplot(1,3,3)
    % show_model(rdnModel.model_def_LC, 'FaceColor', colorModel, 'ColorGCA', colorGCA, ...
    %         'MarkerSize', 20, 'isAnchor', false, 'isLattice', false);
    % title('Deformed Linear Combination')
    
    
    %% Get synthetic data
    numTrain = 5000; % size of the dataset
    tic
    [deltasP, alphas, idxAlphas, idxModel, means, final_data, images_in, poses] = randomSet(GM, numTrain);
    t = toc
    
    params_dir = fullfile(ffd_dir, classUID, 'rendered', 'parameters');
    if ~exist(params_dir)
        mkdir(params_dir);
    end
    
    filename = fullfile(params_dir, 'synthetic_data.mat');
    save(filename, 'deltasP', 'alphas', 'idxAlphas', 'idxModel', 'means', 'final_data')
    
    cd(params_dir)
    h5create('FFD_LC_IDX_params.h5','/GT',[5000 127])
    h5write('FFD_LC_IDX_params.h5', '/GT', final_data)
    
    % save all synthetic images
    h5create('images.h5','/IN',[5000 49152])
    h5write('images.h5', '/IN', images_in)
    cd('C:\dev\img2mesh\matlab\')
    
    save(fullfile(params_dir, 'poses.mat'), 'poses')
    
end
