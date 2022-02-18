using DifferentialEquations
using Plots

function lotka_volterra!(du,u,p,t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end

ul0 = [1.0,1.0]
p = (1.5,1.0,3.0,1.0)
tspanl = (0.0,10.0)

f(u,p,t) = 0.98u
u0 = 1.0
tspan = (0.0,1.0)
prob = ODEProblem(f, u0, tspan)
println(prob)
sol = solve(prob)
plot(sol)





frk(u) = 0.98u

function rk4(fr,u0,dt,num)
    r=zeros(num)
    r[1]=u0
    for i in 2:num
        k1 = fr(u0)
        k2 = fr(u0+dt*k1*0.5)
        k3 = fr(u0+dt*k2*0.5)
        k4 = fr(u0+dt*k3)
        u1 = u0+(dt/6)*(k1+2*k2+2*k3+k4)
        r[i]=u1
        u0=u1
    end
    return r
end

dt = 0.1
t = 0:dt:1.0
num = 10
ur0 = 1.0
r=zeros(num)
r=rk4(frk,u0,dt,num)
println(t,r)
scatter(t,r)
