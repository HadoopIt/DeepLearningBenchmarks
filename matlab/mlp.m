% script using the neural network toolbox for benchmarking a simple mlp/logreg

n_examples = 10000;

bmark = fopen('mlp_matlab.bmark', 'w');

data_x = rand(32,n_examples);
data_y = rand(10,n_examples);

% mlp_32_10
net = newff(data_x, data_y,[],{'softmax'});
init(net);


for i = 1:n_examples
    cs_dx{i} = data_x(:,i);
    cs_dy{i} = data_y(:,i);
end
disp(' small dataset created ...')

net.inputWeights{1,1}.learnFcn = 'learngd';
net.biases{1}.learnFcn = 'learngd';
net.adaptParam.passes = 1;
tic;
net = adapt(net, cs_dx, cs_dy);
t = toc
fprintf(bmark, 'mlp_32_10\t');
fprintf(bmark, 'matlab\t');
fprintf(bmark, '%.2f\n', t);
fprintf(bmark, '# Could not find NLL, using meam square error (the default)\n');

data_x = rand(784, n_examples);
data_y = rand(10,n_examples);

for i = 1: n_examples
    cl_dx{i} = data_x(:,i);
    cl_dy{i} = data_y(:,i);
end

disp(' large dataset created ...')


% mlp_784_10
net = newff(data_x, data_y,[],{'softmax'});
init(net);
net.inputWeights{1,1}.learnFcn = 'learngd';
net.biases{1}.learnFcn = 'learngd';
net.adaptParam.passes = 1;
tic;
net = adapt(net, cl_dx, cl_dy);
t = toc
fprintf(bmark, 'mlp_784_10\t');
fprintf(bmark, 'matlab\t');
fprintf(bmark, '%.2f\n', t);
fprintf(bmark, '# Could not find NLL, using mean square error (the default)\n');

% mlp_784_500_10


net = newff(data_x, data_y,500,{'logsig','softmax'});
init(net);
net.inputWeights{1,1}.learnFcn = 'learngd';
net.biases{1}.learnFcn = 'learngd';
net.adaptParam.passes = 1;
net.layerWeights{2,1}.learnFcn = 'learngd';
tic
net = adapt(net, cl_dx, cl_dy);
t = toc
fprintf(bmark, 'mlp_784_500_10\t');
fprintf(bmark, 'matlab\t');
fprintf(bmark, '%.2f\n', t);
fprintf(bmark, '# Could not find NLL, using mean square error (the default)\n');
%mlp_784_1000_1000_1000_10

net = newff(data_x, data_y,[1000,1000,1000],{'logsig','logsig','logsig','softmax'});
init(net);
net.inputWeights{1,1}.learnFcn = 'learngd';
net.biases{1}.learnFcn = 'learngd';
net.adaptParam.passes = 1;
net.layerWeights{2,1}.learnFcn = 'learngd';
net.layerWeights{3,2}.learnFcn = 'learngd';
net.layerWeights{4,3}.learnFcn = 'learngd';
tic;
net = adapt(net, cl_dx, cl_dy);
t = toc
fprintf(bmark, 'mlp_784_1000_1000_1000_10\t');
fprintf(bmark, 'matlab\t');
fprintf(bmark, '%.2f\n', t);

fprintf(bmark, '# Could not find NLL, using mean square error (the default)\n');
fclose(bmark);
