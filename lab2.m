clc;
clear;

x = 0.1:1/22:1;
% y = (1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x))/2;
d = ((1 + 0.6*sin(2*pi*x/0.7)) + (0.3*sin(2*pi*x))/2);
plot(x, d)

% pirmas pasleptasis sluoksnis
w11_1 = rand(1);
w12_1 = rand(1);
w13_1 = rand(1);
w14_1 = rand(1);
w15_1 = rand(1);

b1_1 = rand(1);
b2_1 = rand(1);
b3_1 = rand(1);
b4_1 = rand(1);
b5_1 = rand(1);

% isejimo sluoksnis
w11_2 = rand(1);
w12_2 = rand(1);
w13_2 = rand(1);
w14_2 = rand(1);
w15_2 = rand(1);

b1_2 = rand(1);

eta = 0.01; % mokymosi tempas

% mokymas
for epoch = 1:50000
    for i = 1:length(x)
        % pirmas pasleptasis sluoksnis
        v1_1 = w11_1 * x(i) + b1_1;
        v2_1 = w12_1 * x(i) + b2_1;
        v3_1 = w13_1 * x(i) + b3_1;
        v4_1 = w14_1 * x(i) + b4_1;
        v5_1 = w15_1 * x(i) + b5_1;

        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);
        y5_1 = tanh(v5_1);

        % isejimo sluoksnis
        v1_2 = w11_2 * y1_1 + w12_2 * y2_1 + w13_2 * y3_1 + w14_2 * y4_1 + w15_2 * y5_1 + b1_2;
        y1_2 = v1_2; % Linear activation for output

        % klaida
        e = d(i) - y1_2;

        % skaiciuojamas klaidos gradientas
        delta1_2 = e;
        delta1_1 = (1 - y1_1^2) * delta1_2 * w11_2;
        delta2_1 = (1 - y2_1^2) * delta1_2 * w12_2;
        delta3_1 = (1 - y3_1^2) * delta1_2 * w13_2;
        delta4_1 = (1 - y4_1^2) * delta1_2 * w14_2;
        delta5_1 = (1 - y5_1^2) * delta1_2 * w15_2;

        % atnaujinam svorius
        % isejimo sluoksnis
        w11_2 = w11_2 + eta * delta1_2 * y1_1;
        w12_2 = w12_2 + eta * delta1_2 * y2_1;
        w13_2 = w13_2 + eta * delta1_2 * y3_1;
        w14_2 = w14_2 + eta * delta1_2 * y4_1;
        w15_2 = w15_2 + eta * delta1_2 * y5_1;
        b1_2 = b1_2 + eta * delta1_2;

        % pasleptasis sluoksnis
        w11_1 = w11_1 + eta * delta1_1 * x(i);
        w12_1 = w12_1 + eta * delta2_1 * x(i);
        w13_1 = w13_1 + eta * delta3_1 * x(i);
        w14_1 = w14_1 + eta * delta4_1 * x(i);
        w15_1 = w15_1 + eta * delta5_1 * x(i);

        b1_1 = b1_1 + eta * delta1_1;
        b2_1 = b2_1 + eta * delta2_1;
        b3_1 = b3_1 + eta * delta3_1;
        b4_1 = b4_1 + eta * delta4_1;
        b5_1 = b5_1 + eta * delta5_1;
    end
end

% Testavimas su MLP
X_test = 0.1:1/220:1;
Y_test = zeros(1, length(X_test));

for i = 1:length(X_test)
    % pirmas pasleptasis sluoksnis
    v1_1_test = w11_1 * X_test(i) + b1_1;
    v2_1_test = w12_1 * X_test(i) + b2_1;
    v3_1_test = w13_1 * X_test(i) + b3_1;
    v4_1_test = w14_1 * X_test(i) + b4_1;
    v5_1_test = w15_1 * X_test(i) + b5_1;

    y1_1_test = tanh(v1_1_test);
    y2_1_test = tanh(v2_1_test);
    y3_1_test = tanh(v3_1_test);
    y4_1_test = tanh(v4_1_test);
    y5_1_test = tanh(v5_1_test);

    % isejimo sluoksnis
    v1_2_test = w11_2 * y1_1_test + w12_2 * y2_1_test + w13_2 * y3_1_test + w14_2 * y4_1_test + w15_2 * y5_1_test + b1_2;
    y1_2_test = v1_2_test; % Linear activation for output

    Y_test(i) = y1_2_test;
end

% Braižymas
hold on;
plot(x, d, 'b', X_test, Y_test, 'g');
legend('Tikrasis', 'MLP Aproksimacija');
title('Daugiasluoksnio perceptrono aproksimacija su 5 neuronais');
xlabel('x');
ylabel('y');
hold off;

