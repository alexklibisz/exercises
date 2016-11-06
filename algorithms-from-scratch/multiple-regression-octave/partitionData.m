function [train,test] = partitionData(D, trainingPercentage)

% determine sizes based on file and percentages.
testPercentage = 1 - trainingPercentage;
nRows = size(D,1);
nTrainRows = round(nRows * trainingPercentage);
nTestRows = round(nRows * testPercentage);

% randomize row numbers.
randRows = randperm(nRows);

% use the randomized row numbers and sizes to partition the data.
train = D(randRows(1:nTrainRows),:);
test = D(randRows(nTrainRows+1:end),:);
end;
