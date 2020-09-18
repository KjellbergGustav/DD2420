function [ msgU, msgD, msgL, msgR ] = updateMessages( msgUPrev, msgDPrev, msgLPrev, msgRPrev, dataCost, lambda )
% Update messages with selected updating algorithm

msgU = zeros(size(dataCost));
msgD = zeros(size(dataCost));
msgL = zeros(size(dataCost));
msgR = zeros(size(dataCost));

[~, ~, nLevels] = size(dataCost);

% circshift shifts the an array/matrix circularly
% 2nd argument indicates the number of shift steps and in which direction
% 3rd argument indicates the axis along the shift is performed
incomingMsgFromU = circshift(msgDPrev, 1, 1);
incomingMsgFromD = circshift(msgUPrev, -1, 1);
incomingMsgFromL = circshift(msgRPrev, 1, 2);
incomingMsgFromR = circshift(msgLPrev, -1, 2);

% Update the mesages using the incoming ones above.
% Use the Potts model for computing the cost \phi(x_i, x_j)

end

