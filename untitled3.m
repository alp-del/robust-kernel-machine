clear all
%rng(1)
augmentation=1;allregu=10.^(-10:1:0);
opt=[1,3,8,1000];sigma=100;perturb=0.05;K=40;
for ii=1:1
    clear accuracy accuracy2 testerror accuracy_robust accuracy2_robust
    [ trainx,trainy,testx,testy] = generate_data(opt);
    trainy0=trainy.*(2*(rand(length(trainy),1)>0)-1);
    tic;
    kernel_train=find_kernel(trainx,trainx,sigma);
    toc
    tic;
    kernel_test=find_kernel(trainx,testx,sigma);
    toc
    for i=1:length(allregu)
        tic;
        temp=real(inv(kernel_train+allregu(i)*size(kernel_train,1)*eye(size(kernel_train,1))));
        toc
        tic;
        predicty=kernel_test'*(temp*trainy0);
        toc
        accuracy(i)=mean((predicty-testy).^2);
        testerror(i)=norm(kernel_train'*(temp*trainy0)-trainy0);
    end
    accuracy
    %augmentation
    if augmentation==1
        trainx=repmat(trainx,K,1);
        temp=randn(size(trainx));
        temp1=sum(temp.^2,2);
        temp0=temp.*repmat(real(temp1.^-0.5),1,size(temp,2));
        trainx=trainx+perturb*temp0;
        trainy=repmat(trainy0,K,1);
        train_n=length(trainy);
        tic;
        kernel_train=find_kernel(trainx,trainx,sigma);
        toc
        tic;
        kernel_test=find_kernel(trainx,testx,sigma);
        toc
        for i=1:length(allregu)
            tic;
            temp=real(inv(kernel_train+size(kernel_train,1)*allregu(i)*eye(size(kernel_train,1))));
            toc
            tic;
            predicty=kernel_test'*(temp*trainy);%predict with lambda=0
            toc
            testerror2(i)=norm(kernel_train'*(temp*trainy)-trainy);
            accuracy2(i)=mean((predicty-testy).^2);
            accuracy2
        end
    end
    allaccuracy(ii,:)=accuracy;
    allaccuracy2(ii,:)=accuracy2;
    mean(allaccuracy-allaccuracy2)
end
