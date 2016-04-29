function theta = normalEquation(X, y)
  theta = pinv(X'*X)*X'* y;
end;
