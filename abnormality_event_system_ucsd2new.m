%   Distribution code Version 1.0 -- Oct 12, 2013 by Cewu Lu 
%
%   The Code is to demo Sparse Combination in our Avenue Dataset, based on the method described in the following paper 
%   [1] "Abnormal Event Detection at 150 FPS in Matlab" , Cewu Lu, Jianping Shi, Jiaya Jia, 
%   International Conference on Computer Vision, (ICCV), 2013
%   
%   The code and the algorithm are for non-commercial use only.
clc;
%% Parameters 
params.H = 160;       % loaded video height size
params.W = 240;       % loaded video width size
params.patchWin = 10; % 3D patch spatial size 
params.tprLen = 5;    % 3D patch temporal length
params.BKH = 16;      % region number in height
params.BKW = 24;      % region number in width
params.srs = 5;       % spatial sampling rate in trainning video volume
params.trs = 2;       % temporal sampling rate in trainning video volume 
params.PCAdim = 100;  % PCA Compression dimension
%params.MT_thr = 5;    % 3D patch selecting threshold 
%params.MT_thr = 0.2;用于自带光流的值
params.MT_thr = 0.8;%用于toolbox 第一次试的 1
%params.smoothness = 0.1;% 自己设定的用于光流平滑的 ucsd1 为1，ucsd2 为0.1
params.threshold = [0.8,0.8,0.8,0.8,0.8,0.8];%根据与摄像头的距离不同设置不同的阈值来划分MHOF

H = params.H;
W = params.W; 
patchWin = params.patchWin;
tprLen = params.tprLen; 
BKH = params.BKH;
BKW = params.BKW;
PCAdim = params.PCAdim;
testFileNum = 12;

addpath('functions')
addpath('data')

%% Training feature generation (about 1 minute)
%  tic;
% fileName = 'data\training_vol(ucsd2)';
% numEachVol = 20000; % The maximum sample number in each training video is 7000 
% trainVolDirs = name_filtering(fileName); 
% %Cmatrix = zeros(tprLen*patchWin^2, length(trainVolDirs)*numEachVol);
% Cmatrix = zeros(80, length(trainVolDirs)*numEachVol);
% rand('state', 0);
% for ii = 1 : length(trainVolDirs)
%     [feaRawTrain, LocV3Train]  = train_features_hoghof([fileName,'\', trainVolDirs{ii}], params);
%     t = randperm(size(feaRawTrain,2));
%     %取10000和获取的立方体的个数的最小值
%     curFeaNum = min(size(feaRawTrain,2),numEachVol);
%     %将提取的立方体的特征存到Cmatrix矩阵
%     Cmatrix(:, numEachVol*(ii - 1) + 1 : numEachVol*(ii - 1) + curFeaNum) =  feaRawTrain(:,t(1:curFeaNum));
%     disp(['Feature extraction in ', num2str(ii),' th training video is done!'])
% end
% Cmatrix(:,sum(abs(Cmatrix)) == 0) = [];

% % % [COEFF,~,latent,~] = princomp(Cmatrix');
% % % cumsum(latent)./sum(latent);
% % % mean_Cmatrix = mean(Cmatrix,2);
% % % %index_bl = find(cumsum(latent)./sum(latent)>0.98);
% % % Tw = COEFF(:,1:PCAdim)';
% % % feaMatPCA = Tw*Cmatrix;  
% % % save('data\sparse_dec\Tw.mat','Tw');
% % % toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%修改部分
param.lamda = 0.1;%for avenue_dataset 0.05-0.1
%
param.alpha = 1;
param.lr1 = 0.05; %for avenue_dataset 0.01
param.lr2 = 0.01;  %for avenue_dataset 0.5
param.m = 200;
%%%%%%%%%%%%%%%%%%[Pf,B] = pred_sparse_dec(feaMatPCA,param);
% feaMatPCA = Cmatrix;
% tic;
% [Pf,B] = pred_sparse_dec_prox(feaMatPCA,param);
% toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %% Sparse combination learning  (about 4 minutes)

%% Testing System 
load('data\sparse_dec\Tw.mat','Tw');
%load('data\sparse_combinations\R.mat','R');
ThrTest = 0.20;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ThrMotionVol = 100;%%%%%%%%%%%%%主意这个地方！！！！！！！10for 自带！！！ 60for toolbox
fileNumAll = 0;
timeAll = 0;
times1 = zeros(testFileNum,1);
times2 = zeros(testFileNum,1);
frames = zeros(size(times1));
for idx = 1 : testFileNum 
    idx 
    load(['data\testing_vol(ucsd2)\vol', sprintf('%.2d',idx), '.mat']); 
    imgVol = im2double(vol);
    %t1 = tic;  
    frames(idx) = size(imgVol,3);
    %%%测试特征提取和推理各自需要的时间
    tic;
    [feaPCA, LocV3] = test_features_hoghof(imgVol, ThrMotionVol, params);
    times1(idx)=toc;
    tic;
    Err = recErrNew(feaPCA,Pf,B,param);
    times2(idx)=toc;
    %%%%%%%%%%%%%Err = recError(feaPCA,R,ThrTest);
    %Err  =smooth(Err,3);
    AbEvent = zeros(BKH, BKW, size(imgVol,3));
    for ii = 1 : length(Err)
        AbEvent(LocV3(1,ii),LocV3(2,ii),LocV3(3,ii)) =  Err(ii);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%AbEvent3 = (AbEvent);
%我们的效果如果不加
AbEvent3 = smooth3( AbEvent, 'box', 5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %t2 = toc(t1);
    save(['data\testing_result\regionalRes_',num2str(idx),'.mat'], 'AbEvent3');
    %fprintf('we can achieve %d FPS in %d th video \n', round(size(imgVol,3)/t2), idx);
    fileNumAll = fileNumAll + size(imgVol,3);
   % timeAll = timeAll + t2;
end
fprintf('average FPS is %d \n', round(fileNumAll/timeAll));

% % video demo
% optThr = 1;
% AbEventShow3 = imgVol; 
% for frameID = 1 : size(imgVol,3)
%     AbEventShow3(:,:,frameID) = double(imresize(AbEvent3(:,:,frameID) ,[H, W],'nearest') > optThr) ;
% end
% for frameID = 1 : size(imgVol,3)  
%     imshow([imgVol(:,:,frameID), AbEventShow3(:,:,frameID)]);
%     getframe;
% end
%% Accuracy result 
optThr = 1.4;
overlapThr = 0.4;
acc = zeros(1, testFileNum);
for idx = 1:testFileNum
    idx
    load(['data\testing_label_mask(ucsd2)\', num2str(idx), '_label.mat'], 'volLabel');
    load(['data\testing_result\regionalRes_',num2str(idx),'.mat'], 'AbEvent3');
    ratios = zeros(1, length(volLabel));
    [Hs, Ws] = size(volLabel{1});
    for ii = 1 : length(volLabel)
        curFrameTemp = double(AbEvent3(:,:,ii) > optThr);
        curFrame = boolean(imresize(curFrameTemp ,[Hs, Ws],'nearest') > 0);
        %curFrame = boolean(imresize(curFrameTemp ,[Hs, Ws],'bilinear') > 0);
        unionSet = sum(sum(curFrame|volLabel{ii}));
        interSet = sum(sum(curFrame&volLabel{ii}));
        if unionSet == 0
            ratios(ii) = 1;
        else
            ratios(ii) = interSet/unionSet;
        end
    end
    acc(idx) = sum(ratios > overlapThr)/length(ratios);
    fprintf('Accuracy in %d th video is %.1f %% \n', idx, 100*acc(idx));
end
fprintf('our overall accuracy is %.1f %% \n', 100*mean(acc));  
























 