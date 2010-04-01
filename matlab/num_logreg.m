n_examples = 6000;

bmark = fopen('clone_numpy_logreg.bmark','w');


data_x = (rand(n_examples,784)-0.5)*1.5;
data_y = (rand(n_examples,10)-0.5)*1.5;
bs = 1;
niter = n_examples/bs;

w = rand(784,10);
b = rand(1,10)*0.;
lr = 0.01;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    hidin = x_i*w + b;
    hidout = (tanh(hidin)+1)/2.;
    g_hidout = hidout - y_i;
    err = 0.5*sum(g_hidout.^2);
    g_hidin = g_hidout .* hidout .* (1. - hidout);

    b = b- lr * sum(g_hidin);
    w = w- lr * x_i' * g_hidin;
end
t = toc

fprintf(bmark, 'mlp_784_10\t');
fprintf(bmark, 'matlab_our_impl{1}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

bs = 60;
niter = n_examples/bs;
b = rand(1,10)*0.;
lr = 0.01;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    hidin = x_i*w + repmat(b, [bs,1]);
    hidout = (tanh(hidin)+1)/2.;
    g_hidout = hidout - y_i;
    err = 0.5*sum(g_hidout.^2);
    g_hidin = g_hidout .* hidout .* (1. - hidout);

    b = b- lr * sum(g_hidin);
    w = w- lr * x_i' * g_hidin;
end
t = toc

fprintf(bmark, 'mlp_784_10\t');
fprintf(bmark, 'matlab_our_impl{60}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

