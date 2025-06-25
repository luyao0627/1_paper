function plot_potential()
u = 0.8; v = 1.3; w = 1.7; k=0.1;
x = -3:0.01:3;
Ux = potential_function(x, u, v, w, k);
U0 = -(u^2*v^2*w^2)/2*x.^2 + (u^2*v^2 + v^2*w^2 + w^2*u^2)/4*x.^4 - ...
    (u^2 + v^2 + w^2)/6*x.^6 + x.^8/8;
figure;
plot(x, Ux, 'b', 'LineWidth', 1.5); hold on;
plot(x, U0, 'c--', 'LineWidth', 1.5);
xlabel('x'); ylabel('U(x)');
title('Original vs Modified Potential Function');
legend('改进势函数 U_x', '原始势函数 U_0');
ylim([-1, 2]); grid on;
end