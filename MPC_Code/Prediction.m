function u_k = Prediction(x_k, E, H, L, R_vec, N, p)
    % MPC控制动作的预测函数（带参考跟踪）
    % x_k: 当前状态
    % E, H, L: MPC设置中的矩阵
    % R_vec: 参考轨迹向量，大小为 (N+1)n x 1
    % N: 控制时域
    % p: 输入数量

    % 计算二次规划问题的线性项系数
    f = 2 * (E' * x_k - L * R_vec);
    
    % 解决二次规划问题
    options = optimset('Display', 'off');
    U_k = quadprog(2*H, f, [], [], [], [], [], [], [], options);

    % 提取第一个控制时段的控制动作(significant)
    u_k = U_k(1:p, 1);
end