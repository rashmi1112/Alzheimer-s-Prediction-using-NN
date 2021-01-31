function M = fillMatrix(Raw_M,mode_educ)

%% Final columns : (1) M/F	(2) Age (3)	EDUC (4) SES 	(5) MMSE 	(6) eTIV 	(7) nWBV 	(8) ASF 
  init =1;
  for i = 2:size(Raw_M,2)-1
    if(find(Raw_M(:,i)==0))
      null_vec(init,1) = i;
    endif
   init += 1; 
  endfor

%% We get all the columns havings null values in them in the null_vec. Now, we need to fill the matrix
%% with values. 
%% Removing the data with null entries would increment our loss proportion of the data, so we need 
%% to analyse each column and fill the values accordingly. 
%% Since, from the data we can see that MMSE has only a few null values, those can be replaced with 
%% median of the whole column. For the columns of SES and EDUC, a lot of values are missing and
%% filling them all with median would approximate our prediction a little too much. So we will 
%% use mode of the column to fill with the most frequently occuring value of the respective column.

median_mmse = median(Raw_M(:,5));
mode_ses = mode(nonzeros(Raw_M(:,4)));

mmse_idx = find(Raw_M(:,5)==0);
Raw_M(mmse_idx,5) = median_mmse;

ses_idx = find(Raw_M(:,4)==0);
Raw_M(ses_idx,4) = mode_ses;

educ_idx = find(Raw_M(:,3)==0);
Raw_M(educ_idx,3) = mode_educ;
  
%% Calculate the correlation between all features and the prediction column(CDR) 
%% to create polnomial features
M = createPolynomialFeatures(Raw_M);
endfunction
