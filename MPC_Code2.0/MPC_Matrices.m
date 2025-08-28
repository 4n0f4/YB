function [E, H, L] = MPC_Matrices(A, B, Q, R_u, F, N)
    % ��ȡϵͳά��
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