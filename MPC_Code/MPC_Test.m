%% ����
clear;
close all;
clc;

% ���ʹ�� MATLAB����ע�͵�����
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

Q = [10 0 ; 0 10];  % Q:״̬����Ȩ�ؾ���n x n ����
F = [10 0 ; 0 10];  % F:�ն����Ȩ�ؾ���n x n ����
R_u = [0.02 0 ; 0 0.02]; % ��������Ȩ�ؾ���p x p ����

k_steps = 50;    % ����step����k

% X_K:״̬����n x k ����
X_K = zeros(n, k_steps);

% ��ʼ״̬����ֵ��n x 1 ����
X_K(:, 1) = [20; -20];

% U_K:�����������p x k ����
U_K = zeros(p, k_steps);

% ����Ԥ������N
N = 10;

% ���� MPC_Matrices ������� E, H, L ����
[E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N);

%% ����ÿһ����״̬������ֵ
for k = 1:k_steps
    % ���ɲο��켣 R
    % �ο�ֵ�� (1,1) ��ʼ��ÿ��ʱ�̵��� 1
    R_vec = [];
    for i = 0:N
        r_val = [k+i; k+i]; % �ο�ֵΪ (k+i, k+i)
        R_vec = [R_vec; r_val];
    end
    
    % ��� U_K(:,k)
    U_K(:, k) = Prediction(X_K(:, k), E, H, L, R_vec, N, p);
    
    % �����k+1��ʱ״̬������ֵ
    X_K(:, k+1) = (A * X_K(:, k) + B * U_K(:, k));
end

%% ����״̬����������ı仯
subplot(2, 1, 1);
hold on;
for i = 1:size(X_K, 1)
    plot(X_K(i, 1:k_steps));
end

% ���Ʋο��켣
plot(1:k_steps, 1:k_steps, '--', 'LineWidth', 1.5);
plot(1:k_steps, 1:k_steps, '--', 'LineWidth', 1.5);
legend("x1", "x2", "R1", "R2");
title('״̬����');
xlabel('ʱ�䲽');
ylabel('ֵ');
hold off;

subplot(2, 1, 2);
hold on;
for i = 1:size(U_K, 1)
    plot(U_K(i, :));
end
legend("u1", "u2");
title('��������');
xlabel('ʱ�䲽');
ylabel('ֵ');
hold off;


