%% 清屏
clear;
close all;
clc;

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

% 权重矩阵
Q = [15 0 ; 0 15];        % 状态权重
R_u = [0.01 0 ; 0 0.01]; % 控制权重  
F = [15 0 ; 0 15];        % 终端权重

k_steps = 100;    % 定义step数量k

% X_K:状态矩阵，n x k 矩阵
X_K = zeros(n, k_steps);

% 初始状态变量值，n x 1 向量
X_K(:, 1) = [20; -20];

% U_K:定义输入矩阵，p x k 矩阵
U_K = zeros(p, k_steps);

% 定义预测区间N
N = 8;

% 调用 MPC_Matrices 函数求得 E, H, L 矩阵
[E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N);

%% 生成随机B样条轨迹
% 设置随机种子以确保可重复性
rng(42);

% 生成随机控制点
num_control_points = 8;
control_points_x = cumsum(randn(1, num_control_points)*5);
control_points_y = cumsum(randn(1, num_control_points)*5);
control_points = [control_points_x; control_points_y];

% 生成B样条轨迹
trajectory = generate_bspline_trajectory_simple(control_points, k_steps + N);

% 提取参考轨迹
R_trajectory = zeros(2, k_steps + N);
for k = 1:k_steps + N
    R_trajectory(:, k) = trajectory(k, :)';
end

%% 计算每一步的状态变量的值
for k = 1:k_steps
    % 提取参考轨迹 (当前时刻到未来N时刻)
    R_vec = [];
    for i = 0:N
        R_vec = [R_vec; R_trajectory(:, k+i)];
    end
    
    % 求得 U_K(:,k)
    U_K(:, k) = Prediction(X_K(:, k), E, H, L, R_vec, N, p);
    
    % 计算第k+1步时状态变量的值
    X_K(:, k+1) = (A * X_K(:, k) + B * U_K(:, k));
end

%% 绘制结果
figure;

% 绘制B样条轨迹和控制点
subplot(2, 2, 1);
plot(control_points(1, :), control_points(2, :), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(trajectory(:, 1), trajectory(:, 2), 'b-', 'LineWidth', 2);
title('B样条轨迹和控制点');
xlabel('X');
ylabel('Y');
legend('控制点', 'B样条轨迹');
grid on;
axis equal;

% 绘制状态变量跟踪结果
subplot(2, 2, 2);
hold on;
plot(1:k_steps, X_K(1, 1:k_steps), 'LineWidth', 1.5);
plot(1:k_steps, X_K(2, 1:k_steps), 'LineWidth', 1.5);
plot(1:k_steps, R_trajectory(1, 1:k_steps), '--', 'LineWidth', 1.5);
plot(1:k_steps, R_trajectory(2, 1:k_steps), '--', 'LineWidth', 1.5);
title('状态变量跟踪');
xlabel('时间步');
ylabel('值');
legend('x1', 'x2', 'R1', 'R2');
grid on;

% 绘制控制输入
subplot(2, 2, 3);
hold on;
plot(1:k_steps, U_K(1, :), 'LineWidth', 1.5);
plot(1:k_steps, U_K(2, :), 'LineWidth', 1.5);
title('控制输入');
xlabel('时间步');
ylabel('值');
legend('u1', 'u2');
grid on;

% 绘制跟踪误差
subplot(2, 2, 4);
tracking_error = sqrt((X_K(1, 1:k_steps) - R_trajectory(1, 1:k_steps)).^2 + ...
                      (X_K(2, 1:k_steps) - R_trajectory(2, 1:k_steps)).^2);
plot(1:k_steps, tracking_error, 'LineWidth', 1.5);
title('跟踪误差');
xlabel('时间步');
ylabel('误差范数');
grid on;

%% 简化的B样条轨迹生成函数
function trajectory = generate_bspline_trajectory_simple(control_points, num_points)
    % 简化的B样条轨迹生成函数
    % control_points: 控制点，2 x n 矩阵
    % num_points: 轨迹点数
    
    n = size(control_points, 2);
    t = linspace(0, n-3, num_points); % 参数t的范围
    
    trajectory = zeros(num_points, 2);
    
    % 使用三次B样条基函数
    for i = 1:num_points
        u = t(i);
        
        % 找到u所在的区间
        k = floor(u) + 1; % 区间索引，从1开始
        
        % 确保k在有效范围内
        k = max(1, min(k, n-3));
        
        % 计算局部参数
        u_local = u - (k-1);
        
        % 三次B样条基函数
        b0 = (1 - u_local)^3 / 6;
        b1 = (3*u_local^3 - 6*u_local^2 + 4) / 6;
        b2 = (-3*u_local^3 + 3*u_local^2 + 3*u_local + 1) / 6;
        b3 = u_local^3 / 6;
        
        % 计算轨迹点
        if k <= n-3
            trajectory(i, :) = b0 * control_points(:, k)' + ...
                               b1 * control_points(:, k+1)' + ...
                               b2 * control_points(:, k+2)' + ...
                               b3 * control_points(:, k+3)';
        else
            % 处理边界情况
            trajectory(i, :) = control_points(:, end)';
        end
    end
end

%% MPC_Matrices 函数
function [E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N)
    n = size(A, 1); % A 是 n x n 矩阵，获取 n
    p = size(B, 2); % B 是 n x p 矩阵，获取 p

    % 初始化 M 矩阵，M 矩阵为 (N+1)n x n 维
    % 顶部为 n x n 单位矩阵
    M = [eye(n); zeros(N*n, n)];

    % 初始化 C 矩阵，初始为 (N+1)n x NP 零矩阵
    C = zeros((N+1)*n, N*p);

    % 定义 M 和 C 矩阵
    tmp = eye(n); % 定义一个 n x n 单位矩阵

    % 更新 M 和 C 矩阵
    for i = 1:N % 循环从 1 到 N
        rows = i*n + (1:n); % 定义当前行，从 i*n 开始，共 n 行
        C(rows, :) = [tmp*B, C(rows-n, 1:end-p)]; % 填充 C 矩阵
        tmp = A * tmp; % 每次将 tmp 乘以 A
        M(rows, :) = tmp; % 填充 M 矩阵
    end

    % 定义 Q_bar 和 R_bar 矩阵
    Q_bar = kron(eye(N), Q);
    Q_bar = blkdiag(Q_bar, F);
    R_bar = kron(eye(N), R_u); 

    % 计算 E, H, L 矩阵
    E = M' * Q_bar * C; % E: n x NP
    H = C' * Q_bar * C + R_bar; % H: NP x NP
    L = C' * Q_bar; % L: NP x (N+1)n

    % 强制 H 为对称矩阵
    H = (H + H') / 2;
end

%% Prediction 函数
function u_k = Prediction(x_k, E, H, L, R_vec, N, p)
    % MPC控制动作的预测函数（带参考跟踪）
    % x_k: 当前状态
    % E, H, L: MPC设置中的矩阵
    % R_vec: 参考轨迹向量，大小为 (N+1)n x 1
    % N: 控制时域
    % p: 输入数量

    % 计算二次规划问题的线性项系数
    f = 2 * (E' * x_k - L * R_vec);
    
    % 解决二次规划问题（不显示优化过程）
    options = optimset('Display', 'off');
    U_k = quadprog(2*H, f, [], [], [], [], [], [], [], options);

    % 提取第一个控制时段的控制动作
    u_k = U_k(1:p, 1);
end


