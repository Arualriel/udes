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



sol   = solve(prob,Tsit5();dt=0.005,adaptive = false,abstol=1e-11,reltol=1e-10)
test_data=Array(solve(prob,Tsit5();dt=0.005,adaptive = false,saveat=t))
plot(sol)
ttrain=4.5f0

sigmoide(x)=1/(1+exp.(-x))
rbf(x)=exp.(-(x.^2))

dudt = FastChain(FastDense(2,9,rbf), FastDense(9,9,sigmoide),FastDense(9,2))


plv = initial_params(dudt)
dudt2_(u,plv,t) = dudt(u,plv)
prob = ODEProblem(dudt2_,u0,(0f0,ttrain),pa)



U=FastChain(FastDense(2,9,rbf), FastDense(9,9,sigmoide),FastDense(9,2))
p=initial_params(U)


function ude_lv(du,u,p,t,pverd)
    ude=U(u,p)
    du[1]=pverd[1]*u[1]+ude[1]
    du[2]=-pverd[4]*u[2]+ude[2]
end

nn_lv(du,u,p,t)=ude_lv(du,u,p,t,pa)

prob_nn=ODEProblem(nn_lv,u0,(0f0,ttrain),p)

function loss(params)
    lprob=remake(prob_nn,p=params)
    predict=Array(solve(lprob,Tsit5();dt=0.005,adaptive = false,saveat=t[t.<=ttrain],abstol=1e-6,reltol=1e-6))
    (sum(abs2,predict-test_data[:,1:size(predict,2)]))/size(predict,2), predict
end


function losslv(p)
  _prob = remake(prob,p=p)
  pred = Array(solve(_prob,Tsit5();dt=0.005,adaptive = false,saveat=t[t .<= ttrain],abstol=1e-6,reltol=1e-6))
  (sum(abs2, pred - test_data[:,1:size(pred,2)])/size(pred,2)),pred
end


iter=0

cb=function (p,l,predict)
    global iter+=1
    if iter%100==0
        @show l
    end
    false
end

iter =-1

res1lv = DiffEqFlux.sciml_train(losslv, plv, ADAM(0.01), maxiters = 1000, cb = cb)
res2lv = DiffEqFlux.sciml_train(losslv, res1lv.minimizer, BFGS(initial_stepnorm=0.01), maxiters = 1000, allow_f_increases=true, cb = cb)

res1=DiffEqFlux.sciml_train(loss,p,ADAM(0.01),maxiters=1000,cb=cb)
res2=DiffEqFlux.sciml_train(loss,res1.minimizer,BFGS(initial_stepnorm=0.01),maxiters=1000,allow_f_increases=true,cb=cb)

ttrain=10f0
prob_nn=ODEProblem(nn_lv,u0,(0f0,ttrain),p)
prob=ODEProblem(dudt2_,u0,(0f0,ttrain),plv)

function loss(params)
    lprob=remake(prob_nn,p=params)
    predict=Array(solve(lprob,Tsit5();dt=0.005,adaptive = false,saveat=t[t.<=ttrain],abstol=1e-6,reltol=1e-6))
    sum(abs2,(predict-test_data[:,1:size(predict,2)])/size(predict,2)), predict
end

function losslv(p)
  _prob = remake(prob,p=p)
  pred = Array(solve(_prob,Tsit5();dt=0.005,adaptive = false,saveat=t[t .<= ttrain],abstol=1e-6,reltol=1e-6))
  (sum(abs2, pred - test_data[:,1:size(pred,2)])/size(pred,2)),pred
end

iter=0

cb=function (p,l,predict)
    global iter+=1
    if iter%100==0
        @show l
    end
    false
end

iter=-1

res3lv = DiffEqFlux.sciml_train(losslv, res2lv.minimizer, BFGS(initial_stepnorm=0.01), maxiters = 1000, allow_f_increases=true, cb = cb)

res3=DiffEqFlux.sciml_train(loss,res2.minimizer,BFGS(initial_stepnorm=0.001),maxiters=1000,allow_f_increases=true,cb=cb)

pl=plot(t,test_data',markersize=5, label=["true x" "true y"])

display(scatter!(pl, t, loss(res3.minimizer)[2]',markersize=5, label=["pred u1 x" "pred u1 y"]))
savefig("ude-rk4-nn.png")

display(scatter!(pl, t, losslv(res3lv.minimizer)[2]',markersize=5, label=["pred u2 x" "pred u2 y"]))
savefig("lvude-rk4-nn.png")

erronn=abs.(losslv(res3lv.minimizer)[2]'-loss(res3.minimizer)[2]')
erroude=abs.(test_data'-loss(res3.minimizer)[2]')
erroudelv=abs.(test_data'-losslv(res3lv.minimizer)[2]')

display(plot(t,erroude,markersize=5, label=["|ude-data| x" "|ude-data| y"]))
savefig("erros-rk4-ude-data-nn.png")

display(plot(t,erroudelv,markersize=5, label=["|udelv-data| x" "|udelv-data| y"]))
savefig("erros-rk4-udelv-data-nn.png")

display(plot(t,erronn,markersize=5, label=["|ude-udelv| x" "|ude-udelv| y"]))
savefig("erros-rk4-nn.png")

