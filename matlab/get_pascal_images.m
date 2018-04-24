add_paths;
data_paths;

%class = 'chair';
%class = 'sofa';
%class = 'bus';
class = 'aeroplane';
%class = 'bicycle';
%class = 'car';
%class = 'motorbike';
%class = 'diningtable';

dir_name = fullfile(pascal_result_dir, class);
dir_pascal = fullfile(PASCAL3D_dir, 'Annotations', [class,'_pascal']);
dir_save_img = fullfile(pascal_images_dir, class);
if ~exist(dir_save_img)
    mkdir(dir_save_img);
end
filenames = dir(dir_name);

maskFac = 1;
images_in = zeros(numel(filenames)-2, 256*192);
fig = figure('Visible', 'off', 'Position', [10000,10000,400,300]); % offscreen figure
set(fig,'color','white')

numCores = 8;
parpool(numCores);
for i = 3:numel(filenames)
    fprintf("%d / %d \n", i-2, numel(filenames));
    file_result = filenames(i).name;
    result = load(fullfile(dir_name, file_result), 'result');
    result = result.result;
    I = result.I.image;

    imshow(I);
    hold on;
    Hmask = imshow(ones(size(I, 1), size(I, 2)));
    Hmask.AlphaData = result.I.segmentation*maskFac;
    I_frame = frame2im(getframe(fig)); % convert plot to image (true color RGB matrix)
       
    I_resized = imresize(I_frame, [192, 256], 'bicubic'); % resize image
    imgfile = fullfile(dir_save_img, [file_result(1:11), '.png']);
    imwrite(I_resized, imgfile); % save image to file
    
    I_gray = rgb2gray(I_resized);
    imshow(I_gray)
    images_in(i-2,:) = I_gray(:);
end

h5create('images_pascal.h5','/IN',[numel(filenames)-2 256*192])
h5write('images_pascal.h5', '/IN', images_in)

fprintf("Done! \n")
