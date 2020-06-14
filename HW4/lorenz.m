clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;
data=[];
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
              r * x(1)-x(1) * x(3) - x(2) ; ...
              x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
for j=1:100  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    data=[data; y];
end
save('lorentz_p28.mat','data','x0', 't')

input=[]; output=[];
for r=[10, 28, 40]
    Lorenz1 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    for j=1:100  % training trajectories
        x0=30*(rand(3,1)-0.5);
        [t,y] = ode45(Lorenz1,t,x0);
        input=[input; y(1:end-1,:)];
        output=[output; y(2:end,:)];
    end
end
save('lorentz_train.mat','input','output','x0', 't')

input=[]; output=[];
for r=[17, 35]
    Lorenz2 = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    for j=1:100  % training trajectories
        x0=30*(rand(3,1)-0.5);
        [t,y] = ode45(Lorenz2,t,x0);
        input=[input; y(1:end-1,:)];
        output=[output; y(2:end,:)];
    end
end
save('lorentz_test.mat','input','output','x0', 't')