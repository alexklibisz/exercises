function mse = meanSquaredError(E, P)
  se = (E - P) .^ 2;
  mse = sum(se) / size(E,1);
end;