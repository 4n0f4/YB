function u_k = Prediction(x_k, E, H, L, R_vec, N, p)
    % MPC���ƶ�����Ԥ�⺯�������ο����٣�
    % x_k: ��ǰ״̬
    % E, H, L: MPC�����еľ���
    % R_vec: �ο��켣��������СΪ (N+1)n x 1
    % N: ����ʱ��
    % p: ��������

    % ������ι滮�����������ϵ��
    f = 2 * (E' * x_k - L * R_vec);
    
    % ������ι滮����
    options = optimset('Display', 'off');
    U_k = quadprog(2*H, f, [], [], [], [], [], [], [], options);

    % ��ȡ��һ������ʱ�εĿ��ƶ���(significant)
    u_k = U_k(1:p, 1);
end