clear;
graphics_toolkit ("gnuplot");

dir = './forest-fires';

% Read in data from csv file.
fileName = sprintf('%s/data-clean.csv', dir);
data = csvread(fileName);

% Split up the data into training and testing.
[trainData, testData] = partitionData(data, 0.70);

% y is the last column of the data.
% x is all but the last column of the data.
trainX = trainData(:,1:end-1);
trainY = trainData(:,end);
testX = testData(:,1:end-1);
testY = testData(:,end);

% Get theta using the training X and Y with the normalEquation
[theta] = normalEquation(trainX, trainY);

% Get the predicted train Y data.
predictedTrain = trainX * theta;
predictedTest = testX * theta;

% Calculate the R^2 and MSE for both sets of predictions.
trainRS = rsquared(predictedTrain, trainY);
trainMSE = meanSquaredError(predictedTrain, trainY);
testRS = rsquared(predictedTest, testY);
testMSE = meanSquaredError(predictedTest, testY);

% Plot the fit for training data
subplot(1,2,1);
x = [0:1:size(trainX,1)-1];
plot(x, trainY, 'b.', 'MarkerSize', 5);
title(sprintf('Train data: m = %d, R^2 = %f, MSE = %f', size(trainData, 1), trainRS, trainMSE));
hold on;
plot(x, predictedTrain, 'r.', 'MarkerSize', 5);
axis([min(x), max(x), min([min(trainY), min(testY)]), max([max(trainY), max(testY)])])
xlabel('Sample');
ylabel('Area burned (ha)');
legend ('actual', 'predicted');

% Plot the fit for test data
subplot(1,2,2);
x = [0:1:size(testX,1)-1];
plot(x, testY, 'b.', 'MarkerSize', 5);
hold on;
plot(x, predictedTest, 'r.', 'MarkerSize', 5);
axis([min(x), max(x), min([min(trainY), min(testY)]), max([max(trainY), max(testY)])])
title(sprintf('Test data: m = %d, R^2 = %f, MSE = %f', size(testData, 1), testRS, testMSE));
xlabel('Sample');
ylabel('Area burned (ha)');
legend ('actual', 'predicted');

% Save plot
p = sprintf('%s/plot.svg',dir);
print(p);