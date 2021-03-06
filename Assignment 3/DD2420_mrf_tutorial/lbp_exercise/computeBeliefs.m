function [ beliefs ] = computeBeliefs(dataCost, msgU, msgD, msgL, msgR)
%   Calculate beliefs based on selected message updating algorithm

beliefs = dataCost;

% Incoming messages for every pixel using circshift
incomingMsgFromU = circshift(msgD, 1, 1);
incomingMsgFromD = circshift(msgU, -1, 1);
incomingMsgFromL = circshift(msgR, 1, 2);
incomingMsgFromR = circshift(msgL, -1, 2);

beliefs = 0;

end

