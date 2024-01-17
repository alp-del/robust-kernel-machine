function [xtrain,ytrain,xtest,ytest]=load_cifar10(a,b,n)
%a=1;
%b=3;
numsample=n;%number of sample
scriptFolder=fileparts(mfilename('fullpath'));
cifar10FolderPath=fullfile(scriptFolder,'cifar-10-batches-mat');

numBatches=5;
allImages=[];
allLabels=[];
for i = 1:numBatches
    batchFileName=fullfile(cifar10FolderPath,['data_batch_',num2str(i),'.mat']);
    data=load(batchFileName);
    allImages=[allImages;data.data];
    allLabels=[allLabels;data.labels];
end
testBatchFileName=fullfile(cifar10FolderPath, 'test_batch.mat');
testData=load(testBatchFileName);
testImages=testData.data;
testLabels=testData.labels;


indicesLabel1=(allLabels==a);% find the indices of data with label=a
trainImagesLabel1=allImages(indicesLabel1,:);
trainlabelsLabel1=allLabels(indicesLabel1,:);

indicesLabel2=(allLabels==b);% find the indices of data with label=b
trainImagesLabel2=allImages(indicesLabel2,:);
trainlabelsLabel2=allLabels(indicesLabel2,:);

trainX=[];
trainY=[];
trainX=[trainImagesLabel1;trainImagesLabel2];
trainY=[trainlabelsLabel1;trainlabelsLabel2];

randomIndices=randperm(size(trainX,1),numsample);
xtrain=trainX(randomIndices,:);
ytrain=trainY(randomIndices,:);

xtrain=im2double(xtrain);
ytrain=im2double(ytrain);

testLabel1=(testLabels==a);% find the indices of data with label=a
testImagesLabel1=testImages(testLabel1,:);
testlabelsLabel1=testLabels(testLabel1,:);

testLabel2=(testLabels==b);% find the indices of data with label=b
testImagesLabel2=testImages(testLabel2,:);
testlabelsLabel2=testLabels(testLabel2,:);

xtest=[testImagesLabel1;testImagesLabel2];
ytest=[testlabelsLabel1;testlabelsLabel2];

xtest=im2double(xtest);
ytest=im2double(ytest);
% Replace labels a with -1 and labels b with 1
ytrain(ytrain == a) = -1;
ytrain(ytrain == b) = 1;

ytest(ytest == a) = -1;
ytest(ytest == b) = 1;
end
