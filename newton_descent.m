function s_opt = newton_descent(s0,t,Q,p,A,b,tol)

% Initialization for current step
s_n = s0;
descent_iters   = 1000;

for k = 1 : descent_iters
    % Computing Derivatives
    
    % Residual Vector
    vector = A*s_n - b;

    % Gradient Value
    g = t * (Q*s_n + p) -  A' * (1./vector);

    % Hessian Value
    h = t*Q;
    for r = 1:size(A,1)
    h = h + (A(r,:)'*A(r,:))./(vector(r)^2);
    end
     
    % Computing Newton Step & Newton Gap
    newton_step   = - h\g;
    % Descent Direction
    newton_direction = - g' * newton_step;
    factor   =  1/(1+sqrt(newton_direction));
    
    % Descent Step Update along Newton Decrement Direction
    s_n1 = s_n + factor * newton_step;  

    s_n = s_n1;
    % Checking Gap vs. Tolerance per descent step
    if  newton_direction/2 < tol
      break;
    end
end
s_opt = s_n;