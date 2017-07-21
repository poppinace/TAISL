function accuracy = normAcc(labels, C)
% normalize the classification accuracy by averaging over different classes

if isnan(C)
  accuracy = 0;
  return
end

clabel = unique(labels);
nclass = length(clabel);
acc = zeros(nclass, 1);
for j = 1:nclass

    c = clabel(j);
    idx = find(labels == c);
    curr_pred_label = C(idx);
    curr_gnd_label = labels(idx);
    acc(j) = length(find(curr_pred_label == curr_gnd_label)) / length(idx);
end

accuracy = mean(acc) * 100;

end