function  MF_vec = labelEncoding(inputArray)
  inputArray1 = cell2mat(inputArray');
  s = size(inputArray1,1);
  MF_vec = size(s,1);
  for i=1:s
    if (inputArray1(i,1)== 'M')
      MF_vec(i,1) = 1;
    else
      MF_vec(i,1) = 0;
    end
  end  
endfunction
