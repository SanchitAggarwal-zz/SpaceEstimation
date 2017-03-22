% Platt’s Probabilistic Outputs for Suppor Vector Machines - Hsuan-Tien Lin et al
% find the coefficients A and B such that the posterior probability
% of P(y=1|x) = 1/(1+exp(A*f(x)+B)), where f(x) is the output
% of the SVM
%
% If no validation set is available, one might use the training
% set to do a leave-one-out procedure. Using the span, this means
% replacing out by something like out-target.*alpha.*span
%
%
% USAGE: [A,B] = getSigmoidParam(out,target)
%
% INPUT:
%       out: vector of outputs of the SVM on a validation set
%       target: validation labels
%       oldA : Previous value of parameter A (default 0)
%       oldB : Previous value of parameter B (default log((prior0+1)/ (prior1+1)))
%           
% OUTPUT:
%       write a model file with struct svmmodel containing model,params,testidx
%       optionaly returns [model,A,B]
%
% Author:Sanchit Aggarwal
% Date:20-september-2013 4:37 A.M.
% Update:18-0ctober-2013 11:14 P.M.

function [A,B] = getSigmoidParam(out,target,oldA,oldB)
    time = tic;
    fnName = 'getSigmoidParam:';
    msg = '----------begins----------';
    fprintf('\n%s %s',fnName,msg);
    % prior1: number of positive points
    prior1 = length(find(target==1));  
    % prior0: number of negative points
    prior0 = length(find(target==-1));
    len = prior0 + prior1;

    A = oldA;
    B = oldB;
    if B == 0
        B = log((prior0+1)/ (prior1+1));
    end
    hiTarget = (prior1+1)/(prior1+2);
    loTarget = 1/(prior0+2);

    lambda = 1e-3;
    olderr = 1e300;
    % temp array to store current estimate of probability of examples
    pp = ones(len,1) * ((prior1+1)/(prior0+prior1+2));

    count = 0;
    for it=1:100
        a=0;
        b=0;
        c=0;
        d=0;
        e=0;
        % Compute the Hessian and gradient of error function
        % with respect to A & B
        for i=1:len
            if(target(i)==1)
                t = hiTarget;
            else
                t = loTarget;
            end
            d1 = pp(i)-t;
            d2 = pp(i) * (1-pp(i));
            a = a + out(i) *out(i) * d2;
            b = b+ d2;
            c = c+ out(i)*d2;
            d = d+ out(i)*d1;
            e = e+d1;
        end
        % If gradient is really tiny then stop
        if( abs(d) < 1e-9 && abs(e) < 1e-9)
            break
        end
        oldA = A;
        oldB = B;
        err = 0;
        % Loop Until goodness of fit increases
        while(1)
            det = (a+lambda) * (b+lambda) - c*c;
            if(det==0)  % if determinant of Hessian is zero
                lambda = lambda*10; % increase stabilizer
                continue;
            end
            A = oldA + ((b+lambda)*d-c*e)/det;
            B = oldB + ((a+lambda)*e-c*d)/det;
            % Now compute the goodness of fit
            err =0 ;
            for i=1:len
                p = 1/(1+exp(out(i)*A+B));
                pp(i) = p;
                % At this step, make sure log(0) returns -200
                err = err - (t*log(p) + (1-t)*log(1-p));
            end
            if(err < olderr*(1+1e-7))
                lambda = lambda*0.1;
                break
            end
            % error did not decrease: increase stabilize by factor of 10 and
            % try again
            lambda = lambda*10;
            if(lambda >= 1e6)  %something is broken Give up
                break
            end
        end
        diff = err - olderr ;
        scale = 0.5 * (err+ olderr+1);
        if(diff > -1e-3*scale && diff < 1e-7*scale)
            count = count+1;
        else
            count = 0;
        end
        olderr = err;
        if(count==3)
            break
        end
    end	
    msg = '----------Finished----------';
    fprintf('\n%s %s',fnName,msg);
    toc(time);
end