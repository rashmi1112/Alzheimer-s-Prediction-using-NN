function M = preprocessData(Long_data,CS_data,MF_vec)

%% Drop columns Subject ID, MRI ID, Group, Visit, MR Delay and Hand column
%% and first row for the labels

Long_colVec = [8,9,10,11,13,14,15];
CS_colVec = [4,5,6,7,9,10,11];

LongValues = Long_data(2:end,Long_colVec);

CSValues = CS_data(2:end,CS_colVec);

%% Some value we will need later.
mode_educ = mode(nonzeros(CSValues(:,3)));

LongValues_Y = Long_data(2:end,12);
CSValues_Y = CS_data(2:end,8);
CDR_Y_temp = [LongValues_Y;CSValues_Y];
CDR_Y = finalY(CDR_Y_temp);


Raw_MX = [LongValues;CSValues];
Raw_MX = [MF_vec, Raw_MX]; 

Raw_M = [Raw_MX,CDR_Y];
size_rawm = size(Raw_M);  %% 9

%% Fill the missing values in the data
M = fillMatrix(Raw_M,mode_educ);
end