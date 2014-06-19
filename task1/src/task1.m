% Exploratory Data Analysis of Abalone Data
% Coded by Max Lotstein based on generated code from
% Neural Pattern Recognition app

% This code reports the test and training error percentages for a 
% a variety of Neural Networks.

% Set to data directory
addpath('C:/Users/Max/Documents/GitHub/ml-freiburg/task1/data/')
% load training data into a table and convert that to a cell array
trainingData = cell2mat(table2cell(readtable('abalone_train.csv', 'Delimiter', ' ')));
randomTrainingOrder = randperm(size(trainingData,1));
trainingData = trainingData(randomTrainingOrder, :);
% use the final 3 columns as the target and the preceding cols as features
% because the toolbox expects the format to be col-oriented, use transpose
trainFeatures = trainingData(:, 1:end-3)';
trainTarget = trainingData(:,end-2:end)';

% do the same for the test data
testData = cell2mat(table2cell(readtable('abalone_test.csv', 'Delimiter', ' ')));
% use the final 3 columns as the target and the preceding cols as features
testFeatures = testData(:, 1:end-3)';
testTarget = testData(:,end-2:end)';

% Create a table to hold the test error as a function of parameters
variableNames = {'NumHiddenLayers','HiddenLayerSize', 'TrainingFn', ....
    'Regularization', 'TrainingError', 'TestError', 'NumberOfEpochs', ...
    'Out1Tar1', 'Out1Tar2', 'Out1Tar3', 'Out2Tar1',...
    'Out2Tar2', 'Out2Tar3', 'Out3Tar1', 'Out3Tar2',...
    'Out3Tar3'};
errorTable = cell2table(cell(0,16));

% Build arrays of variables
numHiddenLayers = [1,2];
hiddenLayerSizes = [5,10,20];
trainingFns = {'traingd','traingda', 'traingdm', 'traingdx', 'trainrp', ...
    'trainbfg', 'trainbr'};
regularizers = [0, .1, .2, .5, 1];
% TODO: introduce trainingFn specific parameters
% set visibility of figures off
set(gcf, 'Visible', 'off')

% Create a neural net object which we can reinitialize during the loop
net = patternnet(1);
% Create a Pattern Recognition Network
for n = numHiddenLayers
    for h = hiddenLayerSizes
        for f = trainingFns
            for r = regularizers
                % create a row vector of hidden layer sizes and initialize
                % the net
                net = patternnet(repmat([h], 1, n));
                
                % Choose Input and Output Pre/Post-Processing Functions
                % For a list of all processing functions type: help nnprocess
                net.input.processFcns = {'removeconstantrows','mapminmax'};
                net.output.processFcns = {'removeconstantrows','mapminmax'};
                
                % Use the training data strictly for training, without
                % reserving any for validation
                net.divideFcn = 'dividetrain';
%                 
%                 net.divideMode = 'sample';  % Divide up every sample
%                 net.divideParam.trainRatio = 70/100;
%                 net.divideParam.valRatio = 15/100;
%                 net.divideParam.testRatio = 15/100;
%                 
                % Set the training function using the iterator
                net.trainFcn = f{1};
                
                % Choose a Performance Function
                % For a list of all performance functions type: help nnperformance
                net.performFcn = 'crossentropy';  % Cross-entropy
                if (r > 0)
                    net.performParam.regularization = r;
                end
                % Choose Plot Functions
                % For a list of all plot functions type: help nnplot
                net.plotFcns = {'plotperform'};

                % Train the Network
                [net,tr] = train(net,trainFeatures,trainTarget);
                % Test the Network
                prediction = net(testFeatures);
                e = gsubtract(testTarget,prediction);
                tind = vec2ind(testTarget);
                yind = vec2ind(prediction);
                percentErrors = sum(tind ~= yind)/numel(tind);
                performance = perform(net,testTarget,prediction);
                % Plots
                % Uncomment these lines to enable various plots.
                perfPlot = figure(1);
                plotperform(tr);
                %saveas(perfPlot, strcat('PerfPlot','[', num2str(n*10), num2str(h*10),f{1}, num2str(r*10),']'), 'png');
                %figure, plottrainstate(tr)
                confPlot = figure(2);
                plotconfusion(testTarget,prediction);
                %saveas(confPlot, strcat('ConfPlot','[', num2str(n*10), num2str(h*10),f{1}, num2str(r*10),']'), 'png');
                %figure, plotroc(t,y)
                %figure, ploterrhist(e)
                
                [c,cm,ind,per] = confusion(testTarget, prediction);
                % Capture the confusion matrix
                cmArray = reshape(cm, 1, 9);
                
                % add line to table
                errorTable = [errorTable; cell2table({n,h,f{1}, r, ...
                    getfield(tr, 'best_perf'), percentErrors, ...
                    max(getfield(tr, 'epoch')), cmArray(1), cmArray(2),...
                    cmArray(3), cmArray(4), cmArray(5), cmArray(6),...
                    cmArray(7), cmArray(8), cmArray(9)})];
            end
        end
    end
end
errorTable.Properties.VariableNames = variableNames;
writetable(errorTable, strcat('errorTable', date ,'.csv'))

% % Recalculate Training, Validation and Test Performance
% trainTargets = t .* tr.trainMask{1};
% valTargets = t  .* tr.valMask{1};
% testTargets = t  .* tr.testMask{1};
% trainPerformance = perform(net,trainTargets,y)
% valPerformance = perform(net,valTargets,y)
% testPerformance = perform(net,testTargets,y)
% 
% % View the Network
% view(net)