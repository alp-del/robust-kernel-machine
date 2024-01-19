[trainx,trainy,testx,testy]=load_cifar10(3,5,500);
delta=8/255; %noise level
delta2=4*delta;
sigma=200;
K=10; %augmentation 
reg=10.^(-12:-1);
temp=repmat(trainx,K,1);
err1=zeros(12,1);
%augmented
for k = 1:length(reg)
    err3=zeros(10,1);
    for i = 1:10
        n=randn(size(temp));
        xtrain=temp+delta*n;
        ytrain=repmat(trainy,K,1);
        ktrain=find_kernel(xtrain,xtrain,sigma);
    
        kinv=inv(ktrain+size(ktrain,1)*reg(k)*eye(size(ktrain,1)));
        err2=zeros(10,1);
    
        for j= 1:10
        
            n2=randn(size(testx));
            xtest=testx+delta2*n2;
            ktest=find_kernel(xtrain,xtest,sigma);
            
            predy=ktest'*(kinv*ytrain);
            
            err2(j)=mean((predy-testy).^2);
        end
        err3(i)=max(err2);
    end
    err1(k)=mean(err3);
end
err1
%no augmentation
error=zeros(12,1);
kernel_train=find_kernel(trainx,trainx,sigma);

for l= 1:length(reg)
    kernelinv=inv(kernel_train+size(kernel_train,1)*reg(l)*eye(size(kernel_train,1)));
    error2=zeros(10,1);
    for i = 1:10
        n3=randn(size(testx));
        xtest=testx+delta2*n3;
    
        
        kerneltest=find_kernel(trainx,xtest,sigma);
        predy2=kerneltest'*(kernelinv*trainy);
        error2(i)=mean((predy2-testy).^2);
    end
    error(l)=max(error2);
end
error

