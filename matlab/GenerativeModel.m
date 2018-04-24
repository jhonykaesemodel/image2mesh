classdef GenerativeModel < handle
    % A generative model of 3D mesh models given an embedding graph.

    properties(GetAccess='public', SetAccess='public')
        classUID;           % UID of class
        fgraph;             % fgraph - embedding graph
        numComp;            % Number of bases
        modelFFD;           % 3D model with FFD
        gmdistFFD;          % Gaussian mixture distribution for the FFD
        MU;                 % Mean of GM distribution for the FFD
        SIGMA;              % Covariance of GM distribution for the FFD
        data_alphas;        % Sparse codes data from Pascal results to fit a distribution
        fitLC;              % Distribution learned from the linear combination parameters
        phi;                % Matrix to impose symmetry on the FFD lattice
        idx_values;         % Indices about the symmetry of the FFD lattice
        idx_zeros;          % Indices about the symmetry of the FFD lattice
        ffdDir;             % FFD directory
        l;                  % Defines the number of FFD control points
        m;                  % Defines the number of FFD control points
        n;                  % Defines the number of FFD control points
    end

    methods
        function obj = GenerativeModel(classUID, SIGMA, fgraph)
            % Construct a free-form deformation (FFD) & linear combination (FFD) 
            % generative model from a 3D embedding graph (fgraph).
            if nargin < 2
                SIGMA = 0.001;
            end
            if nargin < 3
                data_paths;
                graphfile = fullfile(graph_dir, [classUID, '.mat']);
                obj.ffdDir = shapenet_mat_dir;
                pascal_results_dir = fullfile(pascal_result_dir, uid2class(classUID));
                fprintf('Loading LDC graph from %s...\n', graphfile);
                fgraph = load(graphfile);
                fgraph = fgraph.obj;
            end

            obj.classUID = classUID;
            obj.fgraph = fgraph;
            obj.numComp = fgraph.num_nodes;
            obj.l = 3;
            obj.m = 3;
            obj.n = 3;
            obj.SIGMA = SIGMA;
            obj.MU = cell(obj.numComp, 1);
            obj.phi = symmetric_matrix(fgraph.nodes{1}.FFD.P, fgraph.nodes{1}.FFD.l, fgraph.nodes{1}.FFD.m, fgraph.nodes{1}.FFD.n);
            
            % load indices to get only half of the parameters of the FFD lattice
            load('indices_symmetry.mat')
            obj.idx_values = idx_values;
            obj.idx_zeros = idx_zeros;
            
            obj.modelFFD = cell(obj.numComp, 1);
            for i = 1:obj.numComp
                %obj.modelFFD{i} = compute_FFD(fgraph.nodes{i}, obj.l, obj.m, obj.n);
                obj.modelFFD{i} = fgraph.nodes{i}.FFD;
            end
            obj.gmdistFFD = cell(obj.numComp, 1);

            % Generate the GMM for the FFD parameters
            disp('Generating the GMM for the FFD...')
            for source = 1:obj.numComp
                connectedNodes = [1:source-1, source+1:obj.numComp]; % use all nodes
                MU = zeros(numel(connectedNodes), 96);
                for j = 1:numel(connectedNodes)
                    target = connectedNodes(j);
                    ffdFile = fullfile(obj.ffdDir, obj.classUID, sprintf('node%02dto%02d.mat', source, target));
                    fprintf('node%02dto%02d \n', source, target)
                    model = load(ffdFile, 'model');
                    model = model.model;
                    
                    % use only half of the lattice - symmetry constraint
                    vec_dP = vec(model.FFD.dP');
                    vec_dP(idx_zeros) = [];
                    MU(j, :) = vec_dP;               
                                     
                    % get the indices and save it
%                     vec_dP = vec(model.FFD.dP');
%                     vec_dP_full = zeros(192,1);
%                     for i = 1:32
%                          if rem(i, 2) == 0
%                             continue;
%                          end
%                          vec_dP_full(6*i-5:6*i) = vec_dP(6*i-5:6*i);
%                     end     
%                     [idx_zeros, ~] = find(vec_dP_full == 0);
%                     [idx_values, ~] = find(vec_dP_full ~= 0);  
                    
                end
                obj.MU{source} = MU;
                obj.gmdistFFD{source} = gmdistribution(MU, obj.SIGMA*ones(1, 96));
            end
            
            % Fit a normal distribution to the LC (linear combination) parameters (alphas)
            listing = dir(pascal_results_dir);
            data_alphas = zeros(numel(listing)-3, 30);
            disp('Fitting a normal distribution to the linear combination parameters...')
            for i = 3:numel(listing)
                fprintf("%d / %d \n", numel(listing), i);
                load(fullfile(pascal_results_dir, listing(i).name)); 
                
                edges = obj.fgraph.edges(result.c,:);
                [~, idx_alphas] = find(edges == 1);
                                   
                alphas_full = zeros(30,1);
                alphas_full(result.c) = result.sfit.omega(1); % alpha*V + ...
                alphas_full(idx_alphas) = result.sfit.omega(2:end)';
                
                data_alphas(i-2, :) = alphas_full;
            end
            % Fit a distribution using a normal distribution
            obj.fitLC = fitdist(data_alphas(:), 'normal');
            % Visualize the resulting fit
            %index = linspace(min(data_alphas(:)), max(data_alphas(:)), 1000);
            %plot(index, pdf(fitLC, index))    
        end

        function setSIGMA(obj, SIGMA)
            obj.SIGMA = SIGMA;
            for source = 1:obj.numComp
                obj.gmdist{source} = gmdistribution(obj.MU{source}, ...
                    obj.SIGMA*ones(1, 96));
            end           
        end

        function rdnModel = randomModel(obj, modelIndex)
            if nargin < 2
                modelIndex = randi(obj.numComp);
            end
            vec_deltaPt = obj.gmdistFFD{modelIndex}.random(1)';
            
            vec_dP_full = zeros((obj.l*obj.m*obj.n)*3,1); % (4*4*4)x3 = 192
            vec_dP_full(obj.idx_values) = vec_deltaPt;
            vec_dP_full = obj.phi*vec_dP_full;

            [model_def, model_FFD_def] = deform_FFD_lattice(obj.modelFFD{modelIndex}, vec_dP_full);
            
            % take the edges for the linear combination
            edges = obj.fgraph.edges(modelIndex,:);
            [~, idx_alphas] = find(edges == 1);
            numAlphas = numel(idx_alphas);
            % generate random alphas 
            % TODO: better ways to do that?!
            alphas = random(obj.fitLC, numAlphas, 1);
            alphas(alphas <= 0) = 0; 
            alphas(alphas > 0) = 1; % not worried about scale by now.          
            alphas_full = zeros(30,1);
            alphas_full(idx_alphas) = alphas;
            
            % perform the linear combination
            V = model_def.vtx;
            for i = 1:numel(alphas)
                ffdFile = fullfile(obj.ffdDir, obj.classUID, sprintf('node%02dto%02d.mat', modelIndex, idx_alphas(i)));
                model = load(ffdFile, 'model');
                model = model.model;
                
                % figure,
                % show_model(model, 'FaceColor', colorModel, 'ColorGCA', colorGCA, ...
                % 'MarkerSize', 20, 'isAnchor', false, 'isLattice', false);
                
                V = V + alphas(i)*model.vtx;
            end
            model_def_LC.vtx = V;
            model_def_LC.mesh = model_def.mesh;
            
            % data structure
            rdnModel.vec_deltaPt = vec_deltaPt;
            rdnModel.idx_alphas = idx_alphas;
            rdnModel.alphas_full = alphas_full;
            rdnModel.modelIndex = modelIndex;
            rdnModel.model_def = model_def;
            rdnModel.model_def_LC = model_def_LC;
            rdnModel.model_FFD_def = model_FFD_def;    
        end

        function [deltasP, alphas, idxAlphas, idxModel, means, final_data, images_in, poses] = randomSet(obj, numTrain)
            dataDir = fullfile(ffd_dir, obj.classUID);
            if nargin < 2
                trainFile = fullfile(dataDir, 'train.mat');
                trainData = load(trainFile);
                trainData = trainData.voxellist;
                numTrain = size(trainData, 1);
            end

            renderDir = fullfile(dataDir, 'rendered');
            if ~exist(renderDir)
                mkdir(renderDir);
            end
            
            % synthetic image
            dir_save_img = fullfile(renderDir, 'images');
            if ~exist(dir_save_img)
                mkdir(dir_save_img);
            end
            num_poses = 1;
            images_in = zeros(numTrain*num_poses, 256*192);
            poses = cell(numTrain*num_poses,1);
            % synthetic camera
            if isequal(obj.classUID,'02691156')
                [Ri, RiFull] = synthetic_camera(6, 'random');
                projection.scale = 1;
                projection.translation = zeros(2, 1);
            elseif isequal(obj.classUID,'04379243')
                [Ri, RiFull] = synthetic_camera(6, 'vertical');
                projection.scale = 1;
                projection.translation = zeros(2, 1);
            else
                [Ri, RiFull] = synthetic_camera(6, '360'); %360 - random for planes
                projection.scale = 1;
                projection.translation = zeros(2, 1);
            end
                      
            deltasP = [];
            alphas = [];
            idxAlphas = cell(numTrain,1);
            idxModel =[];
            means = [];
            final_data = [];
            fig = figure('Visible', 'off', 'Position', [10000,10000,400,300]); % offscreen figure
            set(fig,'color','white')
            
            parfor i = 1:numTrain
                rdnModel = obj.randomModel();
                filename = fullfile(renderDir, sprintf('render%04d.obj', i));                                            
                
                % normalize model before saving to voxelize later
                mean_vtx = mean(rdnModel.model_def_LC.vtx', 2);
                vtx_norm = bsxfun(@minus, rdnModel.model_def_LC.vtx', mean_vtx);
                rdnModel.model_def_LC.vtx = vtx_norm';
                
                WOBJ(rdnModel.model_def_LC.vtx, rdnModel.model_def_LC.mesh, filename);
                
                deltasP = [deltasP; rdnModel.vec_deltaPt'];
                alphas = [alphas; rdnModel.alphas_full'];
                idxAlphas{i} = rdnModel.idx_alphas; 
                idxModel = [idxModel; rdnModel.modelIndex/100];
                means = [means; mean_vtx'];
                % FFD + alphas + index parameters
                final_data = [final_data; [rdnModel.vec_deltaPt', rdnModel.alphas_full', rdnModel.modelIndex/100]];
                
                % synthetic images
                model = struct();
                model.vertices = rdnModel.model_def_LC.vtx; 
                model.faces = rdnModel.model_def_LC.mesh;
                % fig = figure('Visible', 'off');
                r = randi(6,1,1);
                projection = struct();
                projection.rotation = Ri{r};
                show_model(model, 'projection', projection, 'isAnchor', false, 'lighting', true);
                axis off; 
                I = frame2im(getframe(fig)); % convert plot to image (true color RGB matrix)
                I_resized = imresize(I, [192, 256], 'bicubic'); % resize image
                imgfile = fullfile(dir_save_img, sprintf('img%05d.png', i));
                imwrite(I_resized, imgfile); % save image to file
                I_gray = rgb2gray(I_resized);
                images_in(i,:) = I_gray(:);
                poses{i} = RiFull{r};
                fprintf("Image: %d \n", i);
            end
            close all;
        end
    end
end
