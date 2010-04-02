n_examples = 6000;

bmark = fopen('gpu_our_impl_logreg.bmark','w');


data_x = GPUsingle((rand(n_examples,784)-0.5)*1.5);
data_y = GPUsingle((rand(n_examples,10)-0.5)*1.5);
bs = 1;
niter = n_examples/bs;

w = GPUsingle(rand(784,10));
b = GPUsingle(rand(1,10)*0.);
lr = 0.01;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    hidin = x_i*w + b;
    GPUsync;
    hidout = (tanh(hidin)+1)/2.;
    GPUsync;
    g_hidout = hidout - y_i;
    GPUsync;
    err = 0.5*sum(g_hidout.^2);
    GPUsync;
    g_hidin = g_hidout .* hidout .* (1. - hidout);
    GPUsync;
    b = b- lr * sum(g_hidin);
    GPUsync;
    w = w- lr * x_i' * g_hidin;
    GPUsync;
end
t = toc

fprintf(bmark, 'mlp_784_10\t');
fprintf(bmark, 'matlab_our_impl{g/1}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

bs = 60;
niter = n_examples/bs;
b = GPUsingle(rand(1,10)*0.);
lr = 0.01;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    hidin = x_i*w + repmat(b, [bs,1]);
    GPUsync;
    hidout = (tanh(hidin)+1)/2.;
    GPUsync;
    g_hidout = hidout - y_i;
    GPUsync;
    err = 0.5*sum(g_hidout.^2);
    GPUsync;
    g_hidin = g_hidout .* hidout .* (1. - hidout);
    GPUsync;

    b = b- lr * sum(g_hidin);
    GPUsync;
    w = w- lr * x_i' * g_hidin;
    GPUsync;
end
t = toc

fprintf(bmark, 'mlp_784_10\t');
fprintf(bmark, 'matlab_our_impl{g/60}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

