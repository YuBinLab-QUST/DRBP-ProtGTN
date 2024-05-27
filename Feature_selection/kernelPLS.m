% function [ Xscore ] = kernelPLS(K,Ktest,Y,T,varargin)
% 
% 
% ell = size(K,1);
% trainY = 0;
% KK = K; YY = Y;
% for i=1:T
%     YYK = YY*YY'*KK;
%     beta(:,i) = YY(:,1)/norm(YY(:,1));
%     if size(YY,2) > 1, % only loop if dimension greater than 1
%         bold = beta(:,i) + 1;
%         while norm(beta(:,i) - bold) > 0.001,
%             bold = beta(:,i);
%             tbeta = YYK*beta(:,i);
%             beta(:,i) = tbeta/norm(tbeta);
%         end
%     end
%     tau(:,i) = KK*beta(:,i);
%     val = tau(:,i)'*tau(:,i);
%     c(:,i) = YY'*tau(:,i)/val;
%     trainY = trainY + tau(:,i)*c(:,i)';
%     trainerror = norm(Y - trainY,'fro')/sqrt(ell);
%     %====================================
%    Weight(:,i) = beta(:,i)/norm(tau(:,i));
%    Xscore(:,i) = tau(:,i); % Xscore(:,i) = tau(:,i)/norm(tau(:,i));Ч����
%    %====================================
%     w = KK*tau(:,i)/val;
%     KK = KK - tau(:,i)*w' - w*tau(:,i)' + tau(:,i)*tau(:,i)'*(tau(:,i)'*w)/val;
%     YY = YY - tau(:,i)*c(:,i)';
% end
% 
% % Regression coefficients for new data
% alpha = beta * ((tau'*K*beta)\tau')*Y;
% 
% %  Ktest gives new data inner products as rows, Ytest true outputs
% elltest = size(Ktest',1);
% disp(size(Ktest'));
% disp(size(alpha));
% testY = Ktest' * alpha;
% if ~isempty(varargin)
%     Ytest = varargin{1};
%     testerror = norm(Ytest - testY,'fro')/sqrt(elltest);
%     varargout = testerror;
% end




function [Xscore, alpha, beta] = kernelPLS(K, Y, T)
%kernelPLS 使用核偏最小二乘进行降维
%   输入参数：
%   K - 训练数据的核矩阵
%   Y - 训练数据的目标变量
%   T - 提取的成分数量
%
%   输出参数：
%   Xscore - 降维后的数据
%   alpha - 用于新数据预测的回归系数
%   beta - 成分的权重向量

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
