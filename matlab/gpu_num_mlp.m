function gpu_num_mlp(hostname)

n_examples = 6000;

filename = sprintf('%s_GPU_matlab_mlp.bmark', hostname)
bmark = fopen(filename,'w');

data_x = GPUsingle((rand(n_examples,784)-0.5)*1.5);
data_y = GPUsingle((rand(n_examples,10)-0.5)*1.5);
bs = 1;
niter = n_examples/bs;

w1 = GPUsingle(rand(784,500));
b1 = GPUsingle(rand(1,500)*0.);
w2 = GPUsingle(rand(500,10));
b2 = GPUsingle(rand(1,10)*0.);
lr = 0.01;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    
    hidin = x_i*w1 + b1;
    GPUsync;
    hidout = tanh(hidin);
    GPUsync;
    outin = hidout *w2  + b2;
    GPUsync;
    outout = (tanh(outin)+1)/2.0;
    GPUsync;
    g_outout = outout - y_i;
    GPUsync;
    err = 0.5 * sum(g_outout.^2);
    GPUsync;
    g_outin = g_outout .* outout .* (1.0 - outout);
    GPUsync;
    g_hidout = g_outin * w2';
    GPUsync;
    g_hidin = g_hidout .* (1 - hidout.^2);
    GPUsync;
    b1 = b1 - lr * sum(g_hidin);
    GPUsync;
    b2 = b2 - lr * sum(g_outin);
    GPUsync;
    w1 = w1 - lr * x_i' * g_hidin;
    GPUsync;
    w2 = w2 - lr * hidout' * g_outin;
    GPUsync;

    
  
end
t = toc;

fprintf(bmark, 'mlp_784_500_10\t');
fprintf(bmark, 'matlab{gpu/float/1}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

bs = 60;
niter = n_examples/bs;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    
    hidin = x_i*w1 + repmat(b1, [bs,1]);
    GPUsync;
    hidout = tanh(hidin);
    GPUsync;
    outin = hidout *w2  + repmat(b2, [bs,1]);
    GPUsync;
    outout = (tanh(outin)+1)/2.0;
    GPUsync;
    g_outout = outout - y_i;
    GPUsync;
    err = 0.5 * sum(g_outout.^2);
    GPUsync;
    g_outin = g_outout .* outout .* (1.0 - outout);
    GPUsync;
    g_hidout = g_outin * w2';
    GPUsync;
    g_hidin = g_hidout .* (1 - hidout.^2);
    GPUsync;

    b1 = b1 - lr * sum(g_hidin);
    GPUsync;
    b2 = b2 - lr * sum(g_outin);
    GPUsync;
    w1 = w1 - lr * x_i' * g_hidin;
    GPUsync;
    w2 = w2 - lr * hidout' * g_outin;
    GPUsync;

end
t = toc;

fprintf(bmark, 'mlp_784_500_10\t');
fprintf(bmark, 'matlab{gpu/float/60}\t');
fprintf(bmark, '%.2f\n', n_examples/t);
bs = 1
w1 = GPUsingle(rand(784,1000));
b1 = GPUsingle(rand(1,1000)*0.);
w2 = GPUsingle(rand(1000,1000));
b2 = GPUsingle(rand(1,1000)*0.);
w3 = GPUsingle(rand(1000,1000));
b3 = GPUsingle(rand(1,1000)*0.);
w4 = GPUsingle(rand(1000,10));
b4 = GPUsingle(rand(1,10)*0.);
lr = 0.01;
niter = n_examples/bs;
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    
    hidin1 = x_i*w1 + b1;
    GPUsync;
    hidout1 = tanh(hidin1);
    GPUsync;
    hidin2 = hidout1*w2 + b2;
    GPUsync;
    hidout2 = tanh(hidin2);
    GPUsync;
    hidin3 = hidout2*w3 + b3;
    GPUsync;
    hidout3 = tanh(hidin3);
    GPUsync;
    outin = hidout3 *w4  + b4;
    GPUsync;
    outout = (tanh(outin)+1)/2.0;
    GPUsync;

    g_outout = outout - y_i;
    GPUsync;
    err = 0.5 * sum(g_outout.^2);
    GPUsync;

    g_outin = g_outout .* outout .* (1.0 - outout);
    GPUsync;

    g_hidout3 = g_outin * w4';
    GPUsync;
    g_hidin3 = g_hidout3 .* (1 - hidout3.^2);
    GPUsync;

    g_hidout2 = g_hidin3 * w3';
    GPUsync;
    g_hidin2 = g_hidout2 .* (1 - hidout2.^2);
    GPUsync;

    g_hidout1 = g_hidin2 * w2';
    GPUsync;
    g_hidin1 = g_hidout1 .* (1 - hidout1.^2);
    GPUsync;

    b1 = b1 - lr * sum(g_hidin1);
    GPUsync;
    b2 = b2 - lr * sum(g_hidin2);
    GPUsync;
    b3 = b3 - lr * sum(g_hidin3);
    GPUsync;
    b4 = b4 - lr * sum(g_outin);
    GPUsync;
    w1 = w1 - lr * x_i' * g_hidin1;
    GPUsync;
    w2 = w2 - lr * hidout1' * g_hidin2;
    GPUsync;
    w3 = w3 - lr * hidout2' * g_hidin3;
    GPUsync;
    w4 = w4 - lr * hidout3' * g_outin;
    GPUsync;

    
  
end
t = toc;

fprintf(bmark, 'mlp_784_1000_1000_1000_10\t');
fprintf(bmark, 'matlab{gpu/float/1}\t');
fprintf(bmark, '%.2f\n', n_examples/t);

bs = 60;
niter = n_examples/bs;
w1 = GPUsingle(rand(784,1000));
b1 = GPUsingle(rand(1,1000)*0.);
w2 = GPUsingle(rand(1000,1000));
b2 = GPUsingle(rand(1,1000)*0.);
w3 = GPUsingle(rand(1000,1000));
b3 = GPUsingle(rand(1,1000)*0.);
w4 = GPUsingle(rand(1000,10));
b4 = GPUsingle(rand(1,10)*0.);
tic;
for i = 1:niter
    x_i = data_x((i-1)*bs+1:i*bs,:);
    y_i = data_y((i-1)*bs+1:i*bs,:);
    
    hidin1 = x_i*w1 + repmat(b1, [bs,1]);
    GPUsync;
    hidout1 = tanh(hidin1);
    GPUsync;
    hidin2 = hidout1*w2 + repmat(b2, [bs,1]);
    GPUsync;
    hidout2 = tanh(hidin2);
    GPUsync;
    hidin3 = hidout2*w3 + repmat(b3, [bs,1]);
    GPUsync;
    hidout3 = tanh(hidin3);
    GPUsync;
    outin = hidout3 *w4  + repmat(b4, [bs,1]);
    GPUsync;
    outout = (tanh(outin)+1)/2.0;
    GPUsync;

    g_outout = outout - y_i;
    GPUsync;
    err = 0.5 * sum(g_outout.^2);
    GPUsync;

    g_outin = g_outout .* outout .* (1.0 - outout);
    GPUsync;

    g_hidout3 = g_outin * w4';
    GPUsync;
    g_hidin3 = g_hidout3 .* (1 - hidout3.^2);
    GPUsync;

    g_hidout2 = g_hidin3 * w3';
    GPUsync;
    g_hidin2 = g_hidout2 .* (1 - hidout2.^2);
    GPUsync;

    g_hidout1 = g_hidin2 * w2';
    GPUsync;
    g_hidin1 = g_hidout1 .* (1 - hidout1.^2);
    GPUsync;

    b1 = b1 - lr * sum(g_hidin1);
    GPUsync;
    b2 = b2 - lr * sum(g_hidin2);
    GPUsync;
    b3 = b3 - lr * sum(g_hidin3);
    GPUsync;
    b4 = b4 - lr * sum(g_outin);
    GPUsync;
    w1 = w1 - lr * x_i' * g_hidin1;
    GPUsync;
    w2 = w2 - lr * hidout1' * g_hidin2;
    GPUsync;
    w3 = w3 - lr * hidout2' * g_hidin3;
    GPUsync;
    w4 = w4 - lr * hidout3' * g_outin;
    GPUsync;

    

end
t = toc;

fprintf(bmark, 'mlp_784_1000_1000_1000_10\t');
fprintf(bmark, 'matlab{gpu/float/60}\t');
fprintf(bmark, '%.2f\n', n_examples/t);


fclose(bmark)
exit
end
