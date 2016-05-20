function [Wc,Wg] = imp(I,nRows,nCols,pixels,weights)
Wc = zeros(nRows, nCols, 3, 'double');
for c = 1:3
    Id = double(I(:,:,c))/255;
    Wc(:,:,c) = sum(Id(interpMap.pixels).*interpMap.weights, 3);
end

Wc = min(Wc, 1.0000);
Wc = max(Wc, 0.0000);
Wg = mean(Wc, 3);