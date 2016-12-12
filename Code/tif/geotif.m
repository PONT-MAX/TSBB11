%%
clear
clc
close all

filenames = ['0152945e_582034n_20160905T073402Z_';
    '0152947e_582246n_20160905T073405Z_';
    '0152949e_582459n_20160905T073406Z_';
    '0153357e_582033n_20160905T073402Z_';
    '0153359e_582245n_20160905T073406Z_';
    '0153401e_582458n_20160905T073406Z_';
    '0153403e_582710n_20160905T073405Z_';
    '0153809e_582032n_20160905T073406Z_';
    '0153811e_582244n_20160905T073406Z_';
    '0153813e_582456n_20160905T073406Z_';
    '0153816e_582709n_20160905T073404Z_'
    ]

[rows,cols] = size(filenames)
key.GTModelTypeGeoKey  = 1;  % Projected Coordinate System (PCS)
key.GTRasterTypeGeoKey = 2;  % PixelIsPoint
key.ProjectedCSTypeGeoKey = 32633;
key.ProjLinearUnitsGeoKey = 9001;
key.VerticalUnitsGeoKey = 9001;

for x = 1:rows
    file = filenames(x,:);
    cls_file = ['../auxfiles/',file,'cls.tif'];
    ccls_file = [file,'ccls.tif'];
    png_file = [file,'ccls.png'];
    
    [cls, R] = geotiffread(cls_file); 
    ccls = imread(png_file);
    
    min(min(ccls))
    % Make unknown class #1
    cls(cls ~= 2) = 0;
    cls(cls > 0) = 1;
    
    % Increment class 
    ccls(ccls > 0) = ccls(ccls > 0) + 1;
    cls(ccls > 0) = 0;
    
    % Add to get result
    cls = cls + ccls;
    
    geotiffwrite(ccls_file,ccls,R,'GeoKeyDirectoryTag',key);
end
