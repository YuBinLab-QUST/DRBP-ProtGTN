
function [Xscore, alpha, beta] = kernelPLS(K, Y, T)


ell = size(K,1);
trainY = zeros(size(Y));
Xscore = zeros(ell, T);
beta = zeros(size(K,2), T);
for i=1:T
    YYK = Y*Y'*K;
    beta(:,i) = Y(:,1) / norm(Y(:,1));
    bold = beta(:,i) + 1;
    while norm(beta(:,i) - bold) > 0.001
        bold = beta(:,i);
        tbeta = YYK*beta(:,i);
        beta(:,i) = tbeta / norm(tbeta);
    end
    tau = K*beta(:,i);
    val = tau' * tau;
    c = Y' * tau / val;
    trainY = trainY + tau * c';
    % 存储得分向量
    Xscore(:,i) = tau / norm(tau);
    % 更新K和Y以提取下一个成分
    K = K - tau * (tau' / val);
    Y = Y - tau * c';
end

% 计算回归系数alpha
alpha = beta * (pinv(tau' * K * beta) * tau' * Y);

end
