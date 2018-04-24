function [vertex, faces] = read_obj(filename)

fid = fopen(filename);
if fid<0
    error(['Cannot open ' filename '.']);
end
[str, count] = fread(fid, [1,inf], 'uint8=>char'); 
%fprintf('Read %d characters from %s\n', count, filename);
fclose(fid);

vertex_lines = regexp(str,'v [^\n]*\n', 'match');
vertex = zeros(length(vertex_lines), 3);
for i = 1: length(vertex_lines)
    v = sscanf(vertex_lines{i}, 'v %f %f %f');
    vertex(i, :) = v';
end

face_lines = regexp(str,'f [^\n]*\n', 'match');
faces = zeros(length(face_lines), 3);
for i = 1: length(face_lines)
    f = sscanf(face_lines{i}, 'f %d//%d %d//%d %d//%d');
    if (length(f) == 6) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(3);
        faces(i, 3) = f(5);
        continue
    end
    f = sscanf(face_lines{i}, 'f %d %d %d');
    if (length(f) == 3) % face
        faces(i, :) = f';
        continue
    end
    f = sscanf(face_lines{i}, 'f %d/%d %d/%d %d/%d');
    if (length(f) == 6) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(3);
        faces(i, 3) = f(5);
        continue
    end
    f = sscanf(face_lines{i}, 'f %d/%d/%d %d/%d/%d %d/%d/%d');
    if (length(f) == 9) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(4);
        faces(i, 3) = f(7);
        continue
    end
end

return
