clear all
close all
dsm = imread('../Data/dsm/0153359e_582245n_20160905T073406Z_dsm.tif');
dhm = imread('cp_map.png');
dhm_n = imread('good_height_without_norm_cv2.png');

max(max(dsm))
max(max(dhm))
max(max(dhm_n))

min(min(dsm))
min(min(dhm))
min(min(dhm_n))
%%
dsmc = dhm(500:1500,5600:6400);
%dtmc = dtm(500:1500,5600:6400);

%Uncomment for faster visualization
%dsmc = imresize(dsmc, 0.25);
%dtmc = imresize(dtmc, 0.5);

[x,y]=size(dsmc);
X=1:x;
Y=1:y;
[xx,yy]=meshgrid(Y,X);
%i=im2double(I);
figure;surf(xx,yy,dsmc);