clear all
%rng(1)
robust=0;
augmentation=1;
%opt=[1,3,8,80];sigma=200;perturb=0.01;K=100;allregu=10.^(-17:1:-2);K_robust=50;perturb_robust=0.0001;%
%opt=[1,3,8,400];sigma=784;perturb=0.08;K=40;%
%opt=[2,100,500,50,1];
%opt=[3,100,1000,20];sigma=0.5;perturb=0.01;K=40;allregu=10.^(-10:1:-2);
%opt=[3,50,500,3];sigma=-3;perturb=0.01;K=40;
opt=[4,100,500,40];sigma=opt(4);perturb=0.01;K=40;allregu=10.^(-10:1:-2);
for ii=1:1
    ii
    clear accuracy accuracy2 testerror accuracy_robust accuracy2_robust
    [ trainx,trainy,testx,testy] = generate_data(opt);
    trainy0=trainy.*(2*(rand(length(trainy),1)>0)-1);
    tic;
    kernel_train=find_kernel(trainx,trainx,sigma);
    toc
    %[U,S]=svd(kernel_train);log(diag(S))'
    tic;
    kernel_test=find_kernel(trainx,testx,sigma);
    toc
    if robust==1
        tic;
        %testx_robust=repmat(testx,K_robust,1);
        %temp=randn(size(testx_robust));
        %temp1=sum(temp.^2,2);
        %temp0=temp.*repmat(real(temp1.^-0.5),1,size(temp,2));
        %testx_robust=testx_robust+perturb_robust*temp0;
        %testy_robust=repmat(testy,K_robust,1);
        %kernel_test_robust=find_kernel(trainx,testx_robust,sigma);
        toc
    end
    %tic;[U,S]=svd(kernel_train);S=diag(S);toc
    for i=1:length(allregu)
        tic;
        temp=real(inv(kernel_train+allregu(i)*size(kernel_train,1)*eye(size(kernel_train,1))));
        toc
        tic;
        predicty=kernel_test'*(temp*trainy0);%predict with lambda=0
        toc
        %tic;predicty=(kernel_test'*(U*(diag((S+size(kernel_train,1)*allregu(i)).^-1)*(U'*trainy))));toc
        %accuracy(i)=1-mean(sign(predicty).*testy)/2-0.5;%accuracy of prediction
        accuracy(i)=mean((predicty-testy).^2);
        testerror(i)=norm(kernel_train'*(temp*trainy0)-trainy0);
        if robust==1
            temp1=temp*trainy0;
            temp2=repmat(temp1,1,size(kernel_test,2)).*kernel_test;
            temp3=trainx'*temp2;
            temp4=testx'.*repmat(predicty',size(testx,2),1);
            temp5=temp3-temp4;
            accuracy_robust(i)=max(sum(temp5.^2));


            %predicty_robust=kernel_test_robust'*(temp*trainy);
            %tt=reshape(predicty_robust,length(testy),K_robust);
            %accuracy_robust(i)=max((max(abs(tt-repmat(predicty,1,K_robust))')).^2);

            %tt=reshape(sign(predicty_robust),length(testy),K_robust);
            %tt2=tt.*repmat(testy,1, K_robust);
            %accuracy_robust(i)=mean((predicty_robust-testy_robust).^2);
            %accuracy_robust(i)=1-mean(min(tt2'))/2-0.5;%accuracy of prediction
        end
    end
    accuracy
    %testerror
    if robust==1
        accuracy_robust
    end
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
        if robust==1
        %    tic;
        %    kernel_test_robust=find_kernel(trainx,testx_robust,sigma);
        %    toc
        end

        %tic;[U,S]=svd(kernel_train);toc
        %S=diag(S);toc
        %trainy0=trainy.*(2*rand(length(trainy),1)>0-1);
        for i=1:length(allregu)
            tic;
            temp=real(inv(kernel_train+size(kernel_train,1)*allregu(i)*eye(size(kernel_train,1))));
            toc
            tic;
            predicty2=kernel_test'*(temp*trainy);%predict with lambda=0
            toc
            %tic;
            %[predict_test_adv,predict_train_adv,alle] = adversarial_training(trainx,trainy,K,testx,testy,sigma,0);
            %toc
            %tic;predicty=(kernel_test'*(U*(diag((S+allregu(i)).^-1)*(U'*trainy))));toc
            testerror2(i)=norm(kernel_train'*(temp*trainy)-trainy);
            %accuracy2(i)=1-mean(sign(predicty).*testy)/2-0.5;%accuracy of prediction
            accuracy2(i)=mean((predicty2-testy).^2);
            %accuracy3(i)=mean((predict_test_adv-testy).^2);
            if robust==1
                temp1=temp*trainy;
                temp2=repmat(temp1,1,size(kernel_test,2)).*kernel_test;
                temp3=trainx'*temp2;
                temp4=testx'.*repmat(predicty2',size(testx,2),1);
                temp5=temp3-temp4;
                accuracy2_robust(i)=max(sum(temp5.^2));


                %predicty_robust2=kernel_test_robust'*(temp*trainy);
                %tt=reshape(predicty_robust2,length(testy),K_robust);
                %accuracy2_robust(i)=max((max(abs(tt-repmat(predicty2,1,K_robust))')).^2);

                %tt=reshape(sign(predicty_robust),length(testy),K_robust);
                %tt2=tt.*repmat(testy,1, K_robust);
                %accuracy2_robust(i)=mean((predicty_robust-testy_robust).^2);
                %  accuracy2_robust(i)=1-mean(min(tt2'))/2-0.5;%accuracy of prediction
            end
            accuracy2
            %    testerror2

            if robust==1
                accuracy2_robust
            end
        end
    end
    allaccuracy(ii,:)=accuracy;
    allaccuracy2(ii,:)=accuracy2;
    %allaccuracy3(ii,:)=accuracy3;
    if robust==1
        allaccuracy_robust(ii,:)=accuracy_robust;
        allaccuracy2_robust(ii,:)=accuracy2_robust;
    end
    if robust==0
        mean(allaccuracy-allaccuracy2)
    else
        [log(mean(allaccuracy));log(mean(allaccuracy2));log(mean(allaccuracy_robust));log(mean(allaccuracy2_robust))]
        %[ans(4,:)-ans(2,:);ans(3,:)-ans(1,:)]
    end
end
