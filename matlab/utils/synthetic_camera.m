function [Ri, RiFull] = syntheticCamera(F, type)

if nargin < 2
    type = 'continuous';
end

Ri = cell(F, 1);
RiFull = cell(F, 1);

switch type 
    case 'random'
        for i = 1:F
            R = orth(randn(3));
            Ri{i} = R(2:3, :);
            RiFull{i} = R;
        end
        
    case 'continuous'
        Ux = @(u) [   0, -u(3),  u(2); ... 
                   u(3),     0, -u(1); ...
                  -u(2),  u(1),    0];
        theta = randn(1); % create an angle randomly
        u = randn(3, 1); 
        u = u/norm(u); % create a unit axis randomly
        for i = 1:F
            theta = theta + 1;
            % create a rotation matrix
            orthR = cos(theta)*eye(3) + sin(theta)*Ux(u) + (1-cos(theta))*kron(u,u');
            Ri{i} = orthR(2:3, :);
            RiFull{i} = orthR(1:3, :);
        end
        
    case 'vertical'
        Ux = @(u) [   0, -u(3),  u(2); ... 
                   u(3),     0, -u(1); ...
                  -u(2),  u(1),    0];
        %theta = randn(1);
        theta = 360/F;
        u = [0.2;0;1]; % z-axis rotation 
        for i = 1:F
            % in degrees
            orthR = cosd(theta*i)*eye(3) + sind(theta*i)*Ux(u) + (1-cosd(theta*i))*kron(u,u');
            
            Ri{i} = orthR(2:3, :);
            RiFull{i} = orthR(1:3, :);
        end
              
    case 'real'
        Ux = @(u) [   0, -u(3),  u(2); ...
                   u(3),     0, -u(1); ...
                  -u(2),  u(1),    0];
        %theta = randn(1);
        theta = 2;
        totalRotDeg = 120;
        u = [0;0;1]; % z-axis rotation
        for i = 1:F
            
            orthR = cosd(theta)*eye(3) + sind(theta)*Ux(u) + (1-cosd(theta))*kron(u,u');
            
            if i >= round(F/2) && i <= round((3*F)/4)              
                camActNum = round(F/2) - round((3*F)/4);
                degrees = totalRotDeg / camActNum;
                
                % in degrees
                theta = theta + degrees;
                orthR = cosd(theta)*eye(3) + sind(theta)*Ux(u) + (1-cosd(theta))*kron(u,u');
                
                Ri{i} = orthR(2:3, :);
                RiFull{i} = orthR(1:3, :);
            else
                Ri{i} = orthR(2:3, :);
                RiFull{i} = orthR(1:3, :);
            end
        end
        
        case '360'
        Ux = @(u) [   0, -u(3),  u(2); ... 
                   u(3),     0, -u(1); ...
                  -u(2),  u(1),    0];
        %theta = randn(1);
        theta = 360/F;
        u = [0;0;1]; % z-axis rotation 
        for i = 1:F           
            % in degrees
            orthR = cosd(theta*i)*eye(3) + sind(theta*i)*Ux(u) + (1-cosd(theta*i))*kron(u,u');      
            Ri{i} = orthR(2:3, :);
            RiFull{i} = orthR(1:3, :);
        end
        
    otherwise
        error('%s is a wrong camera type!', type);
end
