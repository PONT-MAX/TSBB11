%%
clear
clc
%%
[x, R] = geotiffread('../../Data/dhm/0153359e_582245n_20160905T073406Z_dhm.tif');
%%
clc
coordRefSysCode = 4326; %4326
%26986, PCS_NAD83_Massachusetts Projected Coordinate System.
%GeoKeyDirectoryTag =
filename = 'map_squares';
image = imread([filename '.png']);
geotiffwrite([filename '.tif'],image,R, ...
    'coordRefSysCode',coordRefSysCode);%, ...
    %'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag);
    
    %'GeoKeyDirectoryTag',
    %GeoKeyDirectoryTag);
