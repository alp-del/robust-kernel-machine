
function k=find_kernel(X,Y,sigma)
if sigma<=0
%     temp=sum(X.^2,2);
%     X=X.*repmat(real(temp.^-0.5),1,size(X,2));
%     temp=sum(Y.^2,2);
%     Y=Y.*repmat(real(temp.^-0.5),1,size(Y,2));

    k=X*Y';
    k=k+abs(sigma)*k.^2;
% elseif sigma==-1
%     k=X*Y';
% elseif sigma==-2
%     temp=sum(X.^2,2);
%     X=X.*repmat(real(temp.^-0.5),1,size(X,2));
%     temp=sum(Y.^2,2);
%     Y=Y.*repmat(real(temp.^-0.5),1,size(Y,2));
%     k=X*Y';
%     k=k.*(pi-acos(k));
% elseif sigma==-4
%     k1=repmat(X,1,size(X,1));k=min(k1,k1');
else
    norms_X=sum(X.^2,2);
    norms_Y=sum(Y.^2,2);
    dis_squared=repmat(norms_X,1,size(Y,1))+(repmat(norms_Y,1,size(X,1)))'-2*X*Y';
    k=exp(-dis_squared/sigma);
end
end