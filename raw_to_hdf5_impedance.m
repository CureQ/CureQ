% Create a function that takes the filepath of the original file, and converts it into the hdf5 format
% Installation of the AxionFileLoader is required for this to work: https://github.com/axionbio/AxionFileLoader

addpath('C:\Users\jveer\OneDrive - UvA\Bureaublad\hogeschool\Afstudeerstage BMT jaar 5\Programmas\AxionFileLoader\');

function raw_to_hdf(filePath)
    % Replace the file extension with .h5 for voltage data and create
    % filepath for impedance data
    hdf5_voltage_path = strrep(filePath, '.raw', '.h5');
    csv_imp_path = strrep(filePath, '.raw', '_impedance.csv')
    disp(hdf5_voltage_path)

    % Load in the MEA data using the Axion file loader
    mea_voltage_data = AxisFile(filePath).RawVoltageData.LoadData;
    mea_voltage_size=size(mea_voltage_data);
    mea_imp_data = AxisFile(filePath).ViabilityImpedanceEvents.ImpedanceValues
    
    % Save the impedance data in a csv file
    writematrix(mea_imp_data, csv_imp_path); 

    % Retrieve dimensions of the voltage data to create an h5 dataset
    allelectrodes=mea_voltage_size(1)*mea_voltage_size(2)*mea_voltage_size(3)*mea_voltage_size(4);
    measurements=size(mea_voltage_data{1,1,1,1}.GetVoltageVector);

    % Create a dataset with the same naming conventions as MCS machines
    % This should allow other tools that read MCS data to read this file
    % as well, given that they don't have other requirements that the file
    % must adhere to
    dataset_name='/Data/Recording_0/AnalogStream/Stream_0/ChannelData';
    % Alternatively, one can change the interal file structure by
    % uncommenting the next piece of code, and altering the folder
    % structure

    %dataset_name='/Data/Rawdata'

    h5create(hdf5_voltage_path,dataset_name,[measurements(1), allelectrodes]);
    h5disp(hdf5_voltage_path);

    % Loop through all the wells from top left to bottom right (for both wells and electrodes, similar to
    % how it works with MCS hdf5 files
    counter=0;
    for verticalwells = 1:mea_voltage_size(1)
        for horizontalwells = 1:mea_voltage_size(2)
            % For some reason, the rows are counted from bottom to top in
            % Axis software, so we do the same here
            for verticalelectrodes = mea_voltage_size(3):-1:1
                for horizontalelectrodes = 1:mea_voltage_size(4)
                    % Axion indexing order: Vertical wells, Horizontal wells,
                    % Horizontal electrodes, Vertical electrodes
                    raw_electrode_data = mea_voltage_data{verticalwells, horizontalwells, horizontalelectrodes, verticalelectrodes}.GetVoltageVector;
                    counter=counter+1;
                    h5write(hdf5_voltage_path, dataset_name, raw_electrode_data, [1, counter], [measurements(1), 1]);

                    % Display progress
                    msg = sprintf(['Converted electrode ', num2str(counter), ' out of ', num2str(allelectrodes)]);
                    fprintf(repmat('\b', 1, length(msg))); % Erase previous message
                    fprintf(msg);
                end
            end
        end
    end
    fprintf('\n');
    disp('Done')
end

% Convert a single file

% Specify the file path here
path= 'C:\PathToData';
raw_to_hdf(path);

% Or convert all files in a folder

% Define the folder where the .raw files are located
% folderPath = 'path/to/mea/data/folder';

% Get a list of all .raw files in the folder
% fileList = dir(fullfile(folderPath, '*.raw'));

% Loop over each file and get the full paths
% for i = 1:length(fileList)
%     % Construct the full file path
%     fullFilePath = fullfile(folderPath, fileList(i).name);
%     raw_to_hdf(fullFilePath);
% end

% This script loops through the axion data from the top left well to the
% bottom right, and from the top left electrode to the bottom right. This
% means it creates an array of wells*electrodes rows and samples amount of
% columns.
% In the case of a 4x6 (24) well with a 4x4 (16) electrode (so 24*16=384)
% configuration  and 3000000 samples, the final array size would be 384*3000000
%
% This script has been tested and verified using a 24 well 16 electrode axion plate
%
% After the conversion has been completed, these voltage files are not compressed yet and might take up a lot of storage space.
% Hdf5 files can be compressed using the following tool: https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html.
% h5repack -f GZIP=1 "input_file" "output_file"
% Depending on the GZIP level, files might shrink up to 6 times in size.
% Besides the voltage data, this script creates a csv file with the
% impedance data. 

