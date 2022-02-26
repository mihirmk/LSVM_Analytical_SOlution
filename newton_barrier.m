function [s_sol,s_hist,tol_gap,obj] = newton_barrier(Q,p,A,b,s0,mu,tol)

% Barrier Method Initializations
t = 1;
total_iters = 1000;
m = size(A,1);
s = s0;
s_hist = s;
obj = 0.5*s0'*Q*s0+p'*s0;
tol_gap = [];

for i = 1 : total_iters
    s =  newton_descent(s,t,Q,p,A,b,tol);
    
    obj  =  [obj 0.5*s'*Q*s + p'*s];
    s_hist  =  [s_hist s];
    tol_gap = [tol_gap m/t];
    
    % Explicit Check on achieved tolerance
    if  (m/t) < tol
        break;
    else
        t  =  mu*t;
    end
end
s_sol  =  s;
fprintf('Barrier Iterations =%d \n',i);