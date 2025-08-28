%% ����
clear;
close all;
clc;

% ���ʹ�� Octave����ȡ��ע�������������� optim ��
% pkg load optim;

%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ��һ��������״̬�ռ����
% ����״̬���� A, n x n ����
A = [1 0.1 ; -1 2];
n = size(A, 1);

% ����������� B, n x p ����
B = [0.2 1 ; 0.5 2];
p = size(B, 2);

% Ȩ�ؾ���
Q = [15 0 ; 0 15];        % ״̬Ȩ��
R_u = [0.01 0 ; 0 0.01]; % ����Ȩ��  
F = [15 0 ; 0 15];        % �ն�Ȩ��

k_steps = 100;    % ����step����k

% X_K:״̬����n x k ����
X_K = zeros(n, k_steps);

% ��ʼ״̬����ֵ��n x 1 ����
X_K(:, 1) = [20; -20];

% U_K:�����������p x k ����
U_K = zeros(p, k_steps);

% ����Ԥ������N
N = 8;

% ���� MPC_Matrices ������� E, H, L ����
[E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N);

%% �������B�����켣
% �������������ȷ�����ظ���
rng(42);

% ����������Ƶ�
num_control_points = 8;
control_points_x = cumsum(randn(1, num_control_points)*5);
control_points_y = cumsum(randn(1, num_control_points)*5);
control_points = [control_points_x; control_points_y];

% ����B�����켣
trajectory = generate_bspline_trajectory_simple(control_points, k_steps + N);

% ��ȡ�ο��켣
R_trajectory = zeros(2, k_steps + N);
for k = 1:k_steps + N
    R_trajectory(:, k) = trajectory(k, :)';
end

%% ����ÿһ����״̬������ֵ
for k = 1:k_steps
    % ��ȡ�ο��켣 (��ǰʱ�̵�δ��Nʱ��)
    R_vec = [];
    for i = 0:N
        R_vec = [R_vec; R_trajectory(:, k+i)];
    end
    
    % ��� U_K(:,k)
    U_K(:, k) = Prediction(X_K(:, k), E, H, L, R_vec, N, p);
    
    % �����k+1��ʱ״̬������ֵ
    X_K(:, k+1) = (A * X_K(:, k) + B * U_K(:, k));
end

%% ���ƽ��
figure;

% ����B�����켣�Ϳ��Ƶ�
subplot(2, 2, 1);
plot(control_points(1, :), control_points(2, :), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(trajectory(:, 1), trajectory(:, 2), 'b-', 'LineWidth', 2);
title('B�����켣�Ϳ��Ƶ�');
xlabel('X');
ylabel('Y');
legend('���Ƶ�', 'B�����켣');
grid on;
axis equal;

% ����״̬�������ٽ��
subplot(2, 2, 2);
hold on;
plot(1:k_steps, X_K(1, 1:k_steps), 'LineWidth', 1.5);
plot(1:k_steps, X_K(2, 1:k_steps), 'LineWidth', 1.5);
plot(1:k_steps, R_trajectory(1, 1:k_steps), '--', 'LineWidth', 1.5);
plot(1:k_steps, R_trajectory(2, 1:k_steps), '--', 'LineWidth', 1.5);
title('״̬��������');
xlabel('ʱ�䲽');
ylabel('ֵ');
legend('x1', 'x2', 'R1', 'R2');
grid on;

% ���ƿ�������
subplot(2, 2, 3);
hold on;
plot(1:k_steps, U_K(1, :), 'LineWidth', 1.5);
plot(1:k_steps, U_K(2, :), 'LineWidth', 1.5);
title('��������');
xlabel('ʱ�䲽');
ylabel('ֵ');
legend('u1', 'u2');
grid on;

% ���Ƹ������
subplot(2, 2, 4);
tracking_error = sqrt((X_K(1, 1:k_steps) - R_trajectory(1, 1:k_steps)).^2 + ...
                      (X_K(2, 1:k_steps) - R_trajectory(2, 1:k_steps)).^2);
plot(1:k_steps, tracking_error, 'LineWidth', 1.5);
title('�������');
xlabel('ʱ�䲽');
ylabel('����');
grid on;

%% �򻯵�B�����켣���ɺ���
function trajectory = generate_bspline_trajectory_simple(control_points, num_points)
    % �򻯵�B�����켣���ɺ���
    % control_points: ���Ƶ㣬2 x n ����
    % num_points: �켣����
    
    n = size(control_points, 2);
    t = linspace(0, n-3, num_points); % ����t�ķ�Χ
    
    trajectory = zeros(num_points, 2);
    
    % ʹ������B����������
    for i = 1:num_points
        u = t(i);
        
        % �ҵ�u���ڵ�����
        k = floor(u) + 1; % ������������1��ʼ
        
        % ȷ��k����Ч��Χ��
        k = max(1, min(k, n-3));
        
        % ����ֲ�����
        u_local = u - (k-1);
        
        % ����B����������
        b0 = (1 - u_local)^3 / 6;
        b1 = (3*u_local^3 - 6*u_local^2 + 4) / 6;
        b2 = (-3*u_local^3 + 3*u_local^2 + 3*u_local + 1) / 6;
        b3 = u_local^3 / 6;
        
        % ����켣��
        if k <= n-3
            trajectory(i, :) = b0 * control_points(:, k)' + ...
                               b1 * control_points(:, k+1)' + ...
                               b2 * control_points(:, k+2)' + ...
                               b3 * control_points(:, k+3)';
        else
            % ����߽����
            trajectory(i, :) = control_points(:, end)';
        end
    end
end

%% MPC_Matrices ����
function [E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N)
    n = size(A, 1); % A �� n x n ���󣬻�ȡ n
    p = size(B, 2); % B �� n x p ���󣬻�ȡ p

    % ��ʼ�� M ����M ����Ϊ (N+1)n x n ά
    % ����Ϊ n x n ��λ����
    M = [eye(n); zeros(N*n, n)];

    % ��ʼ�� C ���󣬳�ʼΪ (N+1)n x NP �����
    C = zeros((N+1)*n, N*p);

    % ���� M �� C ����
    tmp = eye(n); % ����һ�� n x n ��λ����

    % ���� M �� C ����
    for i = 1:N % ѭ���� 1 �� N
        rows = i*n + (1:n); % ���嵱ǰ�У��� i*n ��ʼ���� n ��
        C(rows, :) = [tmp*B, C(rows-n, 1:end-p)]; % ��� C ����
        tmp = A * tmp; % ÿ�ν� tmp ���� A
        M(rows, :) = tmp; % ��� M ����
    end

    % ���� Q_bar �� R_bar ����
    Q_bar = kron(eye(N), Q);
    Q_bar = blkdiag(Q_bar, F);
    R_bar = kron(eye(N), R_u); 

    % ���� E, H, L ����
    E = M' * Q_bar * C; % E: n x NP
    H = C' * Q_bar * C + R_bar; % H: NP x NP
    L = C' * Q_bar; % L: NP x (N+1)n

    % ǿ�� H Ϊ�Գƾ���
    H = (H + H') / 2;
end

%% Prediction ����
function u_k = Prediction(x_k, E, H, L, R_vec, N, p)
    % MPC���ƶ�����Ԥ�⺯�������ο����٣�
    % x_k: ��ǰ״̬
    % E, H, L: MPC�����еľ���
    % R_vec: �ο��켣��������СΪ (N+1)n x 1
    % N: ����ʱ��
    % p: ��������

    % ������ι滮�����������ϵ��
    f = 2 * (E' * x_k - L * R_vec);
    
    % ������ι滮���⣨����ʾ�Ż����̣�
    options = optimset('Display', 'off');
    U_k = quadprog(2*H, f, [], [], [], [], [], [], [], options);

    % ��ȡ��һ������ʱ�εĿ��ƶ���
    u_k = U_k(1:p, 1);
end


