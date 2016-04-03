function [d, train,test] = partitionData(fileName, trainingPercentage, testPercentage)

% read file into matrix d.
d = csvread(fileName); 

% determine sizes based on file and percentages.
nRows = size(d,1);
nTrainRows = round(nRows * trainingPercentage);
nTestRows = round(nRows * testPercentage);

% randomize row numbers.
randRows = randperm(nRows);

% use the randomized row numbers and sizes to partition the data.
train = d(randRows(1:nTrainRows),:);
test = d(randRows(nTrainRows+1:end),:);

