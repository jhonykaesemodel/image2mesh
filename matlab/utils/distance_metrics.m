function [cd, score] = distance_metrics(model_gt, model, theta)

if nargin < 3
    theta = 0.1;
end

if ~isfield(model_gt, 'vtx')
    [model_gt.vtx] = model_gt.vertices;
    model_gt = rmfield(model_gt,'vertices');
    [model_gt.mesh] = model_gt.faces;
    model_gt = rmfield(model_gt,'faces');
end

% compare cad model
% normalize both models
mask = (model_gt.anchor~=0)&(model.anchor~=0);
mean_est = mean(model.vtx(model.anchor(mask), :), 1);
if isequal(size(model_gt.vtx,1), 36) && isequal(numel(model_gt.anchor), 8) % TODO
    model_gt.anchor(2) = 36;
    mean_gtr = mean(model_gt.vtx(model_gt.anchor(mask), :), 1);
else
    mean_gtr = mean(model_gt.vtx(model_gt.anchor(mask), :), 1);
end
model.vtx = bsxfun(@minus, model.vtx, mean_est);
model_gt.vtx = bsxfun(@minus, model_gt.vtx, mean_gtr);

std_est = mean(std(model.vtx(model.anchor(mask), :), 1, 1));
std_gtr = mean(std(model_gt.vtx(model_gt.anchor(mask), :), 1, 1));
model.vtx = model.vtx/std_est;
model_gt.vtx = model_gt.vtx/std_gtr;

R = align_models(model, model_gt);
model.vtx = model.vtx*R';

model = computeMeshInfo(model);
model_gt = computeMeshInfo(model_gt);

% compute distance from model to target
kdtree = KDTreeSearcher(model_gt.vtx);
[U,~] = surfProjection(model, model_gt, kdtree);
dist_model_target = sum(sum((model.vtx - U).^2, 2)) / size(model.vtx, 1);

dist_model_target_theta = sum(sum((model.vtx - U).^2, 2) > theta)...
    / size(model.vtx, 1);

% compute distance from target to model
kdtree = KDTreeSearcher(model.vtx);
[U,~] = surfProjection(model_gt, model, kdtree);
dist_target_model = sum(sum((model_gt.vtx - U).^2, 2)) / size(model_gt.vtx, 1);

dist_target_model_theta = sum(sum((model_gt.vtx - U).^2, 2) > theta) ...
    / size(model_gt.vtx, 1);

cd = dist_model_target + dist_target_model;
score = dist_model_target_theta + dist_target_model_theta;
