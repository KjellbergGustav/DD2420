function [ labels, energy ] = binaryImageDenoising( img, lambda, tau, nIter )
%   Binary image denoising with loopy belief propagation 

dataCost = computeDataCost(img, tau);
energy = zeros(nIter, 1);

[h, w, nLevels] = size(img);

% Messages sent to neighbors in each direction (up, down, left, right)
% Initiliaze messages
msgU = 0;
msgD = 0;
msgL = 0;
msgR = 0;

tStart = tic;
for i = 1:nIter
    
    [msgU, msgD, msgL, msgR] = updateMessages(msgU, msgD, msgL, msgR, dataCost, lambda);
    % Normalize messages
    [msgU, msgD, msgL, msgR] = normalizeMessages(msgU, msgD, msgL, msgR);
    
    % Compute belief's
    beliefs = computeBeliefs(dataCost, msgU, msgD, msgL, msgR);
    
    % Compute labeling of pixels
    labels = computeLabeling(beliefs);
    
    % Compute energy cost
    energy(i) = computeEnergy(dataCost, labels, lambda);
    fprintf('Iteration %i Energy: %3.4f, Time elapsed: %3.3f \n', i, energy(i), toc(tStart))
    
end
fprintf('Algorithm done. \n\n')

end

