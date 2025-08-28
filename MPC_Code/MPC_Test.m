%% 清屏
clear;
close all;
clc;

% 如果使用 MATLAB，则注释掉此行
% 如果使用 Octave，则取消注释以下行来加载 optim 包
% pkg load optim;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 第一步，定义状态空间矩阵
% 定义状态矩阵 A, n x n 矩阵
A = [1 0.1 ; -1 2];
n = size(A, 1);

% 定义输入矩阵 B, n x p 矩阵
B = [0.2 1 ; 0.5 2];
p = size(B, 2);

Q = [10 0 ; 0 10];  % Q:状态变量权重矩阵，n x n 矩阵
F = [10 0 ; 0 10];  % F:终端误差权重矩阵，n x n 矩阵
R_u = [0.02 0 ; 0 0.02]; % 控制输入权重矩阵，p x p 矩阵

k_steps = 50;    % 定义step数量k

% X_K:状态矩阵，n x k 矩阵
X_K = zeros(n, k_steps);

% 初始状态变量值，n x 1 向量
X_K(:, 1) = [20; -20];

% U_K:定义输入矩阵，p x k 矩阵
U_K = zeros(p, k_steps);

% 定义预测区间N
N = 10;

% 调用 MPC_Matrices 函数求得 E, H, L 矩阵
[E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N);

%% 计算每一步的状态变量的值
for k = 1:k_steps
    % 生成参考轨迹 R
    % 参考值从 (1,1) 开始，每个时刻递增 1
    R_vec = [];
    for i = 0:N
        r_val = [k+i; k+i]; % 参考值为 (k+i, k+i)
        R_vec = [R_vec; r_val];
    end
    
    % 求得 U_K(:,k)
    U_K(:, k) = Prediction(X_K(:, k), E, H, L, R_vec, N, p);
    
    % 计算第k+1步时状态变量的值
    X_K(:, k+1) = (A * X_K(:, k) + B * U_K(:, k));
end

%% 绘制状态变量和输入的变化
subplot(2, 1, 1);
hold on;
for i = 1:size(X_K, 1)
    plot(X_K(i, 1:k_steps));
end

% 绘制参考轨迹
plot(1:k_steps, 1:k_steps, '--', 'LineWidth', 1.5);
plot(1:k_steps, 1:k_steps, '--', 'LineWidth', 1.5);
legend("x1", "x2", "R1", "R2");
title('状态变量');
xlabel('时间步');
ylabel('值');
hold off;

subplot(2, 1, 2);
hold on;
for i = 1:size(U_K, 1)
    plot(U_K(i, :));
end
legend("u1", "u2");
title('控制输入');
xlabel('时间步');
ylabel('值');
hold off;


