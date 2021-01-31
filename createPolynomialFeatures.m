function M = createPolynomialFeatures(Raw_M)
Raw_MX = Raw_M(:,1:end-1);
CDR_Y = Raw_M(:,end);
%% Create correlation vector for each feature with CDR vector 
%% Final columns : (1) M/F	(2) Age (3)	EDUC (4) SES 	(5) MMSE 	(6) eTIV 	(7) nWBV 	(8) ASF 

cols = size(Raw_MX,2);
for i = 1:cols 
  corr_vec(i,1) = corr(Raw_MX(:,i),CDR_Y);
end 

%% We get the following values for correlations when run : 
#{
   M/F --> 6.5100e-02 (0.06)
   Age --> 5.6289e-02 (0.05)
   EDUC --> 4.3836e-02  (0.04)
   SES --> 3.2928e-02   (0.03)
   MMSE --> 1.4174e-01  (0.1)
   eTIV --> -8.4124e-03 (-0.008)
   nWBV2 --> 2.6559e-02 (0.02)
   ASF --> 1.7078e-02 (0.01)
#}

Age_sq = Raw_MX(:,2).^2;
Age_cube = Raw_MX(:,2).^3;
MMSE_sq = (Raw_MX(:,5)).^2;
SES_sq = (Raw_MX(:,4)).^2;
SES_cube = (Raw_MX(:,4)).^3;
eTIV_ASF = Raw_MX(:,6) .* (Raw_MX(:,8)); %% Since they have a correlation of -0.99 (high)
Age_SES = Raw_MX(:,2) .* (Raw_MX(:,4));
Age_MMSE = Raw_MX(:,2) .* (Raw_MX(:,5));
Age_EDUC = Raw_MX(:,2) .* (Raw_MX(:,3));

%% Final columns after creating polynomial features would be : 
#{
(1) M/F	
(2) Age
(3)	EDUC 
(4) SES 	
(5) MMSE 	
(6) eTIV 	
(7) nWBV 	
(8) ASF
(9) Age_squared
(10) MMSE_squared
(11) SSE_squared
(12) eTIV * ASF
(13) Age * SES
(14) Age * MMSE
(15) Age * EDUC
#}

M = [Raw_MX, Age_sq, MMSE_sq, SES_sq, eTIV_ASF, Age_SES, Age_MMSE, Age_EDUC, CDR_Y];

size_m_poly = size(M); 
