function [ trainx,trainy,testx,testy] = generate_data(opt)
if opt(1)==1
    load mnist
    a=opt([2,3]);;%digits 2 and 6
    train_n=opt(4);


    ind1=find(test_y(:,a(1))==1);ind2=find(test_y(:,a(2))==1);ind_test=[ind1;ind2];test_n=length(ind_test);%construct the index of test set

    ind1=find(train_y(:,a(1))==1);ind2=find(train_y(:,a(2))==1);ind_train=[ind1;ind2];
    ind_train=sort(ind_train);
    temp=randperm(length(ind_train));
    ind_train=ind_train(temp(1:train_n));%construct the index of train set

    testx=test_x(ind_test,:);%construct test set
    testy=test_y(ind_test,a(1));

    trainx=train_x(ind_train,:);%construct train set
    trainy=train_y(ind_train,a(1));

elseif opt(1)==2
    train_n=opt(2);
    test_n=opt(3);
    p=opt(4);
    separation=opt(5);
    trainy=[ones(train_n/2,1);-1*ones(train_n/2,1)];
    trainx=zeros(train_n,p);
    trainx(1:train_n,1)=randn(train_n,1);
    trainx(1:train_n,1)=trainx(1:train_n,1)+(trainy)*separation;
    trainx((1+train_n/2):train_n,1)=randn(train_n/2,1)+2;
    trainx(:,2:p)=rand(train_n,p-1);

    testy=[ones(test_n/2,1);-1*ones(test_n/2,1)];
    testx=zeros(test_n,p);
    testx(1:test_n,1)=randn(test_n,1);
    testx(1:test_n,1)=testx(1:test_n,1)+(testy)*separation;
    testx(:,2:p)=rand(test_n,p-1);
%    testx=testx+2*randn(size(testx));
    %testy=[ones(test_n/2,1);-1*ones(test_n/2,1)];
elseif opt(1)==3
    train_n=opt(2);
    test_n=opt(3);
    p=opt(4);
    X=randn(train_n+test_n,p);
    temp=sum(X.^2,2);
    X=X.*repmat(real(temp.^-0.5),1,size(X,2));
    trainx=X(1:train_n,:);
    testx=X(train_n+1:end,:);
    temp=generate_function([trainx;testx]);
    trainy=temp(1:train_n,:);
    trainy=trainy+0*randn(size(trainy));
    testy=temp(train_n+1:end,:);
    %testx=testx+0.2*randn(size(testx));
    %trainy=generate_function(trainx);%testy=testy+0.2*randn(size(testy));
elseif opt(1)==4
    train_n=opt(2);
    test_n=opt(3);
    d=opt(4);
    sigma=d;
    kap=exp(0);
    z=zeros(1,d);
    for j=1:d
        z(j)=(1-((j-1)/d)^kap)^(1/kap)
    end
    lambda=diag(z);
    xtemp=randn(train_n+test_n,d);
    x=xtemp*lambda;
    theta=randn(100,d);
    ytemp=find_kernel(x,theta,sigma);
    y=sum(ytemp,2)+0.1^2*randn(train_n+test_n,1);
    trainx=x(1:train_n,:);
    trainy=y(1:train_n,:);
    testx=x(train_n+1:end,:);
    testy=y(train_n+1:end,:);
    
    
        
        

    
end
end

function y=generate_function(X)
y=X(:,1)+X(:,2).^2;
%y=1*X*randn(size(X,2),1)+1*X.^2*randn(size(X,2),1);
%y=mod(floor(15*X(:,1).^2),3);%+(X(:,3).*(X(:,4)+X(:,2))>0)+sin(X(:,4).*X(:,1))+sin(X(:,1)./X(:,3))+X(:,2)*3;

%y=2*(X(:,1).*X(:,2)>0)-(X(:,3).*(X(:,4)+X(:,2))>0)+sin(X(:,4).*X(:,1))+sin(X(:,1)./X(:,3)*1)+X(:,2)*3;
%y=2*(X(:,1).*X(:,2)>0)-(X(:,3).*(X(:,4)+X(:,2))>0)-(X(:,3).*(X(:,4)+X(:,2))<-0.3)+sin(X(:,4).*X(:,1))+sin(X(:,1)./X(:,3)*1)+(3+X(:,2)*3+X(:,2).^2*3).^0.3;
%y=2*sin(X(:,1).^2*250)+X(:,2);
end

