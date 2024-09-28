% Create a function that takes the filepath of the original file, and converts it into the hdf5 format
% Installation of the AxionFileLoader is required for this to work: https://github.com/axionbio/AxionFileLoader

function raw_to_hdf(filePath)
    % Replace the file extension with .h5
    newExtension = '.h5';
    [filePathNoExt, fileName, ~] = fileparts(filePath);
    hdf5_path = fullfile(filePathNoExt, [fileName newExtension]);
    disp(hdf5_path)

    % Load in the MEA data using the Axion file loader
    mea_data = AxisFile(filePath).RawVoltageData.LoadData;
    mea_size=size(mea_data);

    % Retrieve dimensions of the data to create an h5 dataset
    allelectrodes=mea_size(1)*mea_size(2)*mea_size(3)*mea_size(4);
    measurements=size(mea_data{1,1,1,1}.GetVoltageVector);

    % Create a dataset with the same naming conventions as MCS machines
    % This should allow other tools that read MCS data to read this file
    % aswell, given that they don't have other requirements that the file
    % must adhere to
    dataset_name='/Data/Recording_0/AnalogStream/Stream_0/ChannelData';
    % Alternatively, one can change the interal file structure by
    % uncommenting the next piece of code, and altering the folder
    % structure 

    %dataset_name='/Data/Rawdata'

    h5create(hdf5_path,dataset_name,[measurements(1), allelectrodes]);
    h5disp(hdf5_path);
    
    % Loop through all the wells from top left to bottom right (for both wells and electrodes, similar to
    % how it works with MCS hdf5 files
    counter=0;
    for verticalwells = 1:mea_size(1)
        for horizontalwells = 1:mea_size(2)
            % For some reason, the rows are counted from bottom to top in
            % Axis software, so i do the same here
            for verticalelectrodes = mea_size(3):-1:1
                for horizontalelectrodes = 1:mea_size(4)
                    % Axion indexing order: Vertical wells, Horizontal wells,
                    % Horizontal electrodes, Vertical electrodes
                    raw_electrode_data = mea_data{verticalwells, horizontalwells, horizontalelectrodes, verticalelectrodes}.GetVoltageVector;
                    counter=counter+1;
                    h5write(hdf5_path, dataset_name, raw_electrode_data, [1, counter], [measurements(1), 1]);
                    disp(['Converted electrode ', num2str(counter), ' out of ', num2str(allelectrodes)]);
                end
            end
        end
    end
    disp('done')
end

% Specify the file path here
path='G:/MEADATA/tsc/TSC_20DIV.raw';
raw_to_hdf(path);

% This script loops through the axion data from the top left well to the
% bottom right, and from the top left electrode to the bottom right. This
% means it creates an array of wells*electrodes rows and samples amount of
% columns. 
% In the case of a 4*6 (24) well with a 4*4 (16) electrode (so 24*16=384)
% configuration  and 3000000 samples, the array size would be 384*3000000
% 
% This script has been tested and verified using a 24 well 16 electrode axion plate
% 
% After the conversion has been completed, these files are not compressed yet, and might take up a lot of storage space.
% Hdf5 files can be compressed using the following tool: https://support.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack.
% Depending on the GZIP level, files might shrink up to 6 times in size.