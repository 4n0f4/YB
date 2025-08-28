function [E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N)
    % 获取系统维度
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