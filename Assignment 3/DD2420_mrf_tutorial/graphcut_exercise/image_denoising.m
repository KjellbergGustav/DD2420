close all
addpath('maxflow')

% Read original image
I = double(imread('a.png'));
[h,w] = size(I);
% imshow(I, 'InitialMagnification', 'fit')

% Read image with white noise
I1 = double(imread('a2.png'));
% imshow(I1, 'InitialMagnification', 'fit')

% Set unary weights
unary_terms = compute_unary_terms(I1);

% Set edge weights
lambda = 1; % Smoothing parameter
pairwise_terms = compute_pairwise_terms(I1, lambda); 

% Run maxflow
[flow, labels] = maxflow(pairwise_terms, unary_terms);
Ir = reshape(double(labels), h, w);
imshow(Ir)
imwrite(Ir, 'restored_a.png')
