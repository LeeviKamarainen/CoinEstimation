%% Digital Imaging and IMage Pre-Processing
% Taru Haimi & Leevi Kämäräinen
% Practical Assingment, Imaging Measurements

%%
clearvars; close all; clc;



%% Main function:
function [coins] = estim_coins(measurement, bias, dark, flat);
% function [coins] = estim_coins(measurement, bias, dark, flat)
% The function ESTIM_COINS .....
%
% INPUT     measurement     ...
%           bias            ...
%           dark            ...
%           flat            ...
% OUTPUT    [coins]         numbers of coins in the image [5ct, 10ct, 20ct, 50ct, 1e, 2e]

% Inside the function estim_coins you should define two calibrations, first the calibration of the
% intensity and then also the geometric calibration for the input image.
% Typically the output from the geometric calibration is a two-element vector containing the 2D-mapping from millimeters
% to pixels (the first element is corresponding to the horizontal domain and the second element to
% the vertical domain of the image).

end

%% Unzip the files if needed
zipvar = 0;
if zipvar == 1
    untar("DIIP-images-bias.tar")
    untar("DIIP-images-dark.tar")
    untar("DIIP-images-flat.tar")
    untar("DIIP-images-measurements-1.tar")
    untar("DIIP-images-measurements-2.tar")
    addpath("DIIP-images\")
    addpath("DIIP-images\Bias\")
    addpath("DIIP-images\Flat\")
    addpath("DIIP-images\Dark\")
    addpath("DIIP-images\Measurements\")
end

%%
measurementFiles = dir("DIIP-images\Measurements\*.JPG")
flatFiles = dir("DIIP-images\Flat\*.JPG")
darkFiles = dir("DIIP-images\Dark\*.JPG")
biasFiles = dir("DIIP-images\Bias\*.JPG")

for i = 1:1:length(measurementFiles)
    currentFile = measurementFiles(i).name;
    currentImage = imread(currentFile);
    measurementImages{i} = currentImage;
end

for i = 1:1:length(flatFiles)
    currentFile = flatFiles(i).name;
    currentImage = imread(currentFile);
    flatImages{i} = currentImage;
end

for i = 1:1:length(darkFiles)
    currentFile = darkFiles(i).name;
    currentImage = imread(currentFile);
    darkImages{i} = currentImage;
end

for i = 1:1:length(biasFiles)
    currentFile = biasFiles(i).name;
    currentImage = imread(currentFile);
    biasImages{i} = currentImage;
end


%% Practical assignment
%% DIIP

clearvars;close all;clc



%% Unzip the files if needed
zipvar = 0;
if zipvar == 1
    untar("DIIP-images-bias.tar")
    untar("DIIP-images-dark.tar")
    untar("DIIP-images-flat.tar")
    untar("DIIP-images-measurements-1.tar")
    untar("DIIP-images-measurements-2.tar")
end

    addpath("DIIP-images\")
    addpath("DIIP-images\Bias\")
    addpath("DIIP-images\Flat\")
    addpath("DIIP-images\Dark\")
    addpath("DIIP-images\Measurements\")
%%

measurementFiles = dir("DIIP-images\Measurements\*.JPG")
flatFiles = dir("DIIP-images\Flat\*.JPG")
darkFiles = dir("DIIP-images\Dark\*.JPG")
biasFiles = dir("DIIP-images\Bias\*.JPG")
% % 
% for i = 1:1:length(measurementFiles)
%     currentFile = measurementFiles(i).name;
%     currentImage = imread(currentFile);
%     measurementImages{i} = currentImage;
% end
% 
% for i = 1:1:length(flatFiles)
%     currentFile = flatFiles(i).name;
%     currentImage = imread(currentFile);
%     flatImages{i} = currentImage;
% end
% 
% for i = 1:1:length(darkFiles)
%     currentFile = darkFiles(i).name;
%     currentImage = imread(currentFile);
%     darkImages{i} = currentImage;
% end
% 
% for i = 1:1:length(biasFiles)
%     currentFile = biasFiles(i).name;
%     currentImage = imread(currentFile);
%     biasImages{i} = currentImage;
% end


% Mean images:
flatImage = 0;
darkImage = 0;
biasImage = 0;
for i = 1:1:length(measurementFiles)
    currentFile = measurementFiles(i).name;
    currentImage = imread(currentFile);
    measurementImages{i} = currentImage;
end

for i = 1:1:length(flatFiles)
    currentFile = flatFiles(i).name;
    currentImage = double(imread(currentFile));
    flatImage = flatImage+currentImage;
end

for i = 1:1:length(darkFiles)
    currentFile = darkFiles(i).name;
    currentImage = double(imread(currentFile));
    darkImage = darkImage+currentImage;
end

for i = 1:1:length(biasFiles)
    currentFile = biasFiles(i).name;
    currentImage = double(imread(currentFile));
    biasImage = biasImage+currentImage;
end

% flatImage = flatImage./length(flatFiles);
darkImage = darkImage./length(darkFiles);
biasImage = biasImage./length(biasFiles);

%%
ind = 8;
estim_coins(measurementImages{ind},biasImage,darkImage,flatImage)
%%
%%
function [coins] = estim_coins(measurement, bias, dark, flat);
% function [coins] = estim_coins(measurement, bias, dark, flat)
% The function ESTIM_COINS .....
%
% INPUT     measurement     The raw image that have been measured.
%           bias            Mean bias image composed of all of the bias images.
%           dark            Mean dark image composed of all of the dark images.
%           flat            Mean flat image composed of all of the flat images.
%
% OUTPUT    [coins]         numbers of coins in the image [5ct, 10ct, 20ct, 50ct, 1e, 2e]


%Show the original image:
    subplot(4,2,1)
    imshow(measurement); title("Original image")

    %Intensity calibration process:
    rawimg = uint8(measurement); % Raw image
    flatfield = uint8(flat); %Flatfield images summed together
    bmean = uint8(bias); %Mean bias images

    dmean = uint8(dark)-bmean; % Mean dark images
    flatfield = flatfield-bmean-dmean; %Calculate the flatfield image
    flatfieldnorm = uint8(double(flatfield)./(double(max(flatfield(:))))); %Normalise the flatfield image using the max value.
    
    %Calibrated image:
    calibratedImg = (rawimg-bmean-dmean)./flatfieldnorm;
    %Show the calibrated image:
    subplot(4,2,2)
    imshow(uint8(calibratedImg)); title("Intensity calibrated image")
    
    %Geometric calibration:
    %Grayscale the image:
    subplot(4,2,3)
    grayImg = im2gray(calibratedImg);
    imshow(grayImg); title("Gray scale image")
    %Finding the checkerboard points from the rawimage.
    [cbPoints, boardSize] = detectCheckerboardPoints(rawimg);

    %Using two consecutive checkerboard points, we can calculate what is
    %distance (in pixels
    cbDist = sqrt((cbPoints(1,1)-cbPoints(2,1))^2+(cbPoints(1,2)-cbPoints(2,2))^2);
    cbPixel = cbDist;
    %Then because we know that one square is 12.5 mm wide, we can calculate
    %the width of single pixel in millimeters.
    pixelL = 12.5/cbPixel;


    %Binarization of the grayscale image:
    %Edge command tries to find the edges of grayscale image.
    BW = edge(grayImg); 
    subplot(4,2,4)
    imshow(BW); title("Binarized image")

    %Dilatate the image to create more defined edges:
    %Help from: https://se.mathworks.com/help/images/ref/imdilate.html
    %We use disk shaped structuring element with radius of 25 for the
    %dilatation and erosion.
    morphParam = 25;
    se = strel('disk',morphParam);
    dilatedI = imdilate(BW,se);
    subplot(4,2,5)
    imshow(dilatedI); title("Dilated image (creating edges)")

    %Fill the circles using imfill function:
    filledI = imfill(dilatedI,'holes');
    subplot(4,2,6)
    imshow(filledI); title("Filling shapes")

    %Erode the squares using the structuring element defined before.:
    erodedI = imerode(filledI,se);
    subplot(4,2,7)
    imshow(erodedI); title("Eroded image")

    %Find the circles in the image. We try to find all of the circles with
    %radiuses between 190 and 300, and we use the sensitivity of 0.965 to
    %guarantee that we find all of the circles.
    [centers, radii] = imfindcircles(erodedI,[190 300],'ObjectPolarity','bright','Sensitivity',0.97);
    subplot(4,2,8)
    imshow(erodedI); title("Finding coins in the image")
    hold on
    coins = find_coins(radii,pixelL);
    viscircles(centers,radii);
    sgtitle('Image processing steps') ;

end

%%
function [foundCoins] = find_coins(calcRadii, pixelLength)
% function [foundCoins] = find_coins(calcRadii, pixelLength)
%
% The function FIND_COINS makes comparison to which known coin radii the
% calculated radii matches best, and based on that it decides which coins
% and how many each of them are found in the image.
%
% INPUT     calcRadii       the radii of found coins in the image     
%           pixelLength     how wide one pixel is in mm. 
% OUTPUT    [foundCoins]    numbers of coins in the image [5ct, 10ct, 20ct, 50ct, 1e, 2e]

% Known radii of coins [5ct, 10ct, 20ct, 50ct, 1e, 2e]:
coinRadii = [21.25 19.75 22.25 24.25 23.25 25.75]./2; % mm
coinradiipx = coinRadii./pixelLength; % px
foundCoins = zeros(1,6);

for i = 1:length(calcRadii)
    % Finding how close the calculated radius is to the known coin radii,
    % and selecting the coin with smallest difference:
    coin = calcRadii(i);
    dists = abs(coin-coinradiipx);
    minDistCoin = find(dists==min(dists));

    % Updating the number of found coins:
    foundCoins(minDistCoin) = foundCoins(minDistCoin) + 1;

end