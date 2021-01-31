function W = randInitializeTheta(L_in, L_out)
  W = zeros(L_out, 1 + L_in);
  E_INIT = sqrt(6)/ sqrt(L_in + L_out);
  W = rand(L_out,1+L_in) * (2*E_INIT) - E_INIT;
endfunction
