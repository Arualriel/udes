import Pkg
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("PyPlot")

using DifferentialEquations

f(u,p,t) = 0.98u
u0 = 1.0
tspan = (0.0,1.0)
prob = ODEProblem(f, u0, tspan)
println(prob)

sol = solve(prob)

using Plots
pyplot()

plot(sol)

plot(sol, linewidth=3, title="Solução para a EDO linear", xaxis="Tempo (t)", yaxis="u(t)", label="Solução Obtida")

plot!(sol.t, t->1.0*exp(0.98t), lw=3, ls=:dash, label="Solução Exata")

sol.t

sol.u

sol.t[3]

sol.u[3]

[t+u for (u,t) in tuples(sol)]

sol(0.33)

sol = solve(prob, abstol=1e-8, reltol=1e-8)

plot(sol)
plot!(sol.t, t->1.0*exp(0.98t), lw=3, ls=:dash, label="Solução Exata")

sol = solve(prob, saveat=0.1)

sol = solve(prob, saveat=[0.35,0.71,0.99])

sol = solve(prob, saveat=[0.35,0.71,0.99], save_start=false, save_end=false)

sol = solve(prob, dense=false)

sol = solve(prob, save_everystep=false)

sol = solve(prob, alg_hints=[:stiff])

sol = solve(prob, Tsit5(), reltol=1e-6)

function lorentz!(du,u,p,t)
    s,r,b = p
    du[1] = s*(u[2]-u[1])
    du[2] = u[1]*(r-u[3])-u[2]
    du[3] = u[1]*u[2]-b*u[3]
end

u0 = [1.0,0.0,0.0]

p = (10,28,8/3)
tspan = (0.0,100.0)
prob = ODEProblem(lorentz!, u0, tspan, p)
sol = solve(prob)

sol.t[10],sol.u[10]

sol[3,5]

A = convert(Array, sol)

plot(sol)

plot(sol, vars=(1,2,3))

plot(sol, vars=(1,2,3), denseplot=false)

plot(sol, vars=(0,3))

function lotka_volterra!(du,u,p,t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end

lv! = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d

u0 = [1.0,1.0]
p = (1.5,1.0,3.0,1.0)
tspan = (0.0,10.0)
prob = ODEProblem(lv!,u0,tspan,p)
sol = solve(prob)
plot(sol)

A = [1. 0  0 -5
     4 -2  4 -3
    -4  0  0  1
     5 -2  2  3]
u0 = rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)

sol[3]

big_u0 = big.(u0)

prob = ODEProblem(f,big_u0,tspan)
sol = solve(prob)

sol[2,3]

prob = ODEProblem(f,big_u0,big.(tspan))
sol = solve(prob)

Pkg.pkg"add StaticArrays"
using StaticArrays
A = @SMatrix [ 1.0  0.0 0.0 -5.0
               4.0 -2.0 4.0 -3.0
              -4.0  0.0 0.0  1.0
               5.0 -2.0 2.0  3.0]
u0 = @SMatrix rand(4,2)
tspan = (0.0,1.0)
f(u,p,t) = A*u
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)

sol[3]


