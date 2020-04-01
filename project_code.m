dataDir= '\Data';
checkpointDir = 'modelCheckpoints';

rng(1)
Symmetry_Groups = {'Abstract_Expressionism', 'Art_Nouveau_Modern', 'Baroque' ,'Color_Field_Painting', 'Cubism',... 
    'Expressionism', 'Impressionism', 'Naive_Art_Primitivism', 'Northern_Renaissance',...
    'Pop_Art', 'Post_Impressionism', 'Rococo', 'Romanticism', 'Symbolism'};

train_folder = 'train';
test_folder  = 'test';

fprintf('Loading Train Filenames and Label Data...');
t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);

[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));


fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));


rng('default');
numEpochs = 20; 
batchSize = 100;
nTraining = length(train.Labels);

layers = [
    imageInputLayer([224 224 3]); 
    
    convolution2dLayer(3,16,'Padding',[1 1],'Stride', [1,1]);  
    convolution2dLayer(3,16,'Padding',[1 1],'Stride', [1,1]);  
    reluLayer(); 
    maxPooling2dLayer(2,'Stride',2); 
    
    convolution2dLayer(3,32,'Padding',[1 1],'Stride', [1,1]); 
    convolution2dLayer(3,32,'Padding',[1 1],'Stride', [1,1]);   
    reluLayer(); 
    maxPooling2dLayer(2,'Stride',2); 
    
    convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]);   
    convolution2dLayer(3,64,'Padding',[1 1],'Stride', [1,1]); 
    reluLayer(); 
    maxPooling2dLayer(2,'Stride',2); 
      
    fullyConnectedLayer(25); 
    dropoutLayer(.25); 
    fullyConnectedLayer(14); 
    softmaxLayer();
    classificationLayer(); 
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',1e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
   

 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

YTest = classify(net1,val);
val_acc = mean(YTest==val.Labels)

plotTrainingAccuracy_All(info1,numEpochs);


YTest = classify(net1,test);
test_acc = mean(YTest==test.Labels)

%Salincy map for one image
img=train.Files{4536};
img=imread(img);
map = simpsal(img);
figure(1)
imshow(img);
figure(2)
mapbig = mat2gray(imresize( map , [ size(img,1) size(img,2) ] )); 
imshow(mapbig);
