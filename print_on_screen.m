function print_on_screen(acc, annotations)

method = {
  'NA    '
  'NTSL  '
  'TAISL '
};
        
fprintf('--------------------------------------\n')
fprintf([annotations.prm.sourceName '-->' annotations.prm.targetName '\n'])
for i = 1:length(method)
  fprintf( ...
    ['acccuracy - ' method{i} ' = %3.1f(%3.1f)\n'], ...
    acc{i}{1}, ...
    acc{i}{2} ...
  )
end
fprintf('--------------------------------------\n')