function CDR_Y = finalY(CDR_Y_temp)

%% Replace the label 0 with Label 10 & Label 0.5 with 50, since octave does not have 0 indices
zero_vec = find(CDR_Y_temp==0);
half_vec = find(CDR_Y_temp==0.5);
CDR_Y_temp(zero_vec,1) = 3;
CDR_Y_temp(half_vec,1) = 4;
CDR_Y = CDR_Y_temp;
end