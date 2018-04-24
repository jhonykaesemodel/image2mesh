function WOBJ(vertices, faces, filename)

fid = fopen(filename, 'W');
fprintf(fid, 'v %.6f %.6f %.6f\n', vertices');
fprintf(fid, 'f %d %d %d\n', faces');
fclose('all');

