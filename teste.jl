

using OrdinaryDiffEq
using Plots
using DiffEqFlux, Optim
using Flux: ADAM

function lotkavolterra(du,u,p,t)
    a,b,c,d=p
    du[1]=a*u[1]-b*u[1]*u[2]
    du[2]=-c*u[2]+d*u[1]*u[2]
end

u0= Float32[0.5,1.5]
T= Float32.((0.0,10.0))
pa= Float32[1.25,0.7,1.1,0.9]
datasize=100
t=range(T[1],T[2],length=datasize)
prob=ODEProblem(lotkavolterra,u0,T,pa)



sol=solve(prob,Vern7(),abstol=1e-11,reltol=1e-10,saveat=T)
test_data=Array(solve(prob,Vern7(),saveat=T))
plot(sol)
ttrain=2.5f0

sigmoide(x)=1/(1+exp.(-x))
rbf(x)=exp.(-(x.^2))
U=FastChain(FastDense(2,9,rbf), FastDense(9,9,sigmoide),
    FastDense(9,9,relu),FastDense(9,2))
p=initial_params(U)


function ude_lv(du,u,p,t,pverd)
    ude=U(u,p)
    du[1]=pverd[1]*u[1]+ude[1]
    du[2]=-pverd[4]*u[2]+ude[2]
end

nn_lv(du,u,p,t)=ude_lv(du,u,p,t,pa)

prob_nn=ODEProblem(nn_lv,u0,(0f0,ttrain),p)

function pred(params,cinicial=u0,tempo=T)
    Array(solve(prob_nn,Vern7(),cinicial,p=params,tspan=(0f0,ttrain),
            saveat=tempo[tempo.<=ttrain],abstol=1e-6,reltol=1e-6))
end

function loss(params)
    predict=pred(params)
    (sum(abs2,predict-test_data[:,1:size(pred,2)]))/size(pred,2)
end

iter=0

cb=function (p,l,pred)
    global iter+=1
    if iter%10==0
        @show l
    end
    false
end

iter =-1

res1= DiffEqFlux.sciml_train(loss,p,ADAM(0.01),maxiters=100,cb=cb)
res2=DiffEqFlux.sciml_train(loss, res1.minimizer,
    BFGS(initial_stepnorm=0.01),maxiters=100,allow_f_increases=true,cb=cb)

ttrain=10f0
prob=ODEProblem(nn_lv,u0,(0f0,ttrain),p)

function pred(params,cinicial=u0,tempo=T)
    Array(solve(prob_nn,Vern7(),cinicial,p=params,tspan=(0f0,ttrain),
            saveat=tempo[tempo.<=ttrain],abstol=1e-6,reltol=1e-6))
end

function loss(params)
    predict=pred(params)
    (sum(abs2,predict-test_data[:,1:size(pred,2)]))/size(pred,2)
end

iter=0

cb=function (p,l,pred)
    global iter+=1
    if iter%10==0
        @show l
    end
    false
end

iter=-1

res3=DiffEqFlux.sciml_train(loss,res2.minimizer,
    BFGD(initial_stepnorm=0.001),maxiters=100,allow_f_increases=true,cb=cb)
plot(t,test_data',markersize=5, label=["true x" "true y"])
scatter!(pl, t, loss(res3.minimizer)[2]',markersize=5, label=["pred x" "pred y"])
