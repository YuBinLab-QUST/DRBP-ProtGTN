% Step 1: Load the dataset
X = csvread('PDB14189hunhe2.csv', 1, 0); % Skipping the header row if present
% Step 2: Construct Y based on your criteria
Y = [ones(7129, 1); zeros(7060, 1)]; % Assuming 1 for positive and 0 for negative

% Optionally, if you want to use binary encoding for Y
% Though not necessary for binary classification in this context
% Y = binarize(Y);

% Step 3: Data Normalization
[X_normalized, ~, meanX, stdX] = normalizemeanstd(X);

% Step 4: Compute the Kernel Matrix
% Example using a polynomial kernel
type = 'polynomial';
par1 = 2; % Degree of the polynomial
coef = 1; % Offset for the polynomial kernel
K = kernel(X_normalized, X_normalized, type, par1, coef);

% Step 5: KPLS Dimensionality Reduction
num_Component = 100; % Desired number of dimensions
[kplsXS] = kernelPLS(K, Y, num_Component);

% Now kplsXS contains the reduced dimensionality data.
csvwrite('PDB14189_KPLS_21.csv', kplsXS);