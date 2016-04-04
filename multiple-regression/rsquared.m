% E and P should be (m x 1) matrices.
function r = rsquared(E, P)

  % subtracting leaves 0's on any equal values for E and P.
  D = E - P;
  % number of zeros is the number of equal values for E and P.
  zeroCount = sum(D(:)==0);
  % r-squared is the percentage of equal values
  r = zeroCount / size(E, 1);

end;
