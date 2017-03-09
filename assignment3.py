##Script for Assignment 3##
"""
Christopher Thompson, 140185164
"""
import numpy as np
from matplotlib import pyplot as plt

def midpoint(tmin,tmax,n):
    """The midpoint method applied to solving equation (2), our equation for damped harmonic motion.
    args:
        tmin: the smallest time value in your time-domain.
        tmax: the largest time value in your time domain.
        n: number of steps you would like to take in your time-domain.
        
    returns:
        ts: array containing n+1 time values 
        us: array containing n+1 midpoint approximations to u(t)
        vs: array containing n+1 midpoint approximations to v(t)
        h: size of interval between any two time values in ts
    """
    ts=np.linspace(tmin,tmax,n+1)
    h=(tmax - tmin)/n #calculates step size
    us=np.zeros(n+1); vs=np.zeros(n+1) #initialize two arrays with n+1 values
    us[0] = 1.0; vs[0]=0.0 #sets initial values of u and v
    for k in range(n):
        u1=us[k] + 0.5*h*vs[k] # midpoint estimate
        v1=vs[k] - 0.5*h*((2*vs[k])+(5*us[k]))
        us[k+1] = us[k] + h*v1 #calculates midpoint approximation of u for the k+1 interval
        vs[k+1] = vs[k] - h*(2*v1+5*u1) #calculates midpoint approximation of v for the k+1 interval
    return ts, us, vs, h

def error_func(tmin,tmax,nb=1,nt=21,nstep=1,t=2):
    """Calculates the error between the midpoint approximations of u and v, and the analytical solutions of u and v at a given time t for a range of h values.
    args:
        tmin: the smallest time value in your time-domain.
        tmax: the largest time value in your time domain.
        nb: lowest value used to calculate range of n values in midpoint function.
        nt: highest value used to calculate range of n values in midpoint function
        nstep: step-size used to calculate range of n values in narray.
        t: the required time t value at which you would like to interpolate u and v from the midpoint method.
    returns:
        h_array: array of h values used in the midpoint method, useful for plotting graph of h values against error values.
        error_2: array of error values calculated in the function, error defined as in assignment brief, see line 54.
    """
    u_exact_t=np.exp(-t)*(np.cos(2*t)+0.5*np.sin(2*t)) #calculates exact value of u at time t
    v_exact_t= np.exp(-t)*-2.5*np.sin(2*t) ##calculates exact value of v at time t
    narray=100*np.arange(nb,nt,nstep) #array of n values used to calculate step-size h
    error_2=[] #initialize error list
    h_array=[] ##initialize h value list
    for i in narray:
        ts,us,vs,h=midpoint(0.0,4.0,i) #calculates midpoint approximations for ith narray value as n
        u_interp=np.interp(t,ts,us) #interpolates the value of u at time t from the midpoint method data ts and us
        v_interp=np.interp(t,ts,vs)#interpolates the value of v at time t from the midpoint method data ts and vs
        error_2.append(np.sqrt((((u_interp)-(u_exact_t))**2)+(((v_interp)-(v_exact_t))**2))) #calculate error 
        h_array.append(h)
    return h_array, error_2

def midpoint2(tmin,tmax,n,a,b):
    """Numerical approximation of the siff ODE in part 2 of the assignment brief of u and v over time t by the midpoint method
    args:
        tmin: the smallest time value in your time-domain.
        tmax: the largest time value in your time domain.
        a: a value used in ODEs
        b: b value used in ODEs
    returns:
        ts: array containing n+1 time values 
        us: array containing n+1 midpoint approximations to u(t)
        vs: array containing n+1 midpoint approximations to v(t)
        h: size of interval between any two time values in ts
    """
    ts=np.linspace(tmin,tmax,n+1)
    h=(tmax-tmin)/n
    us=np.zeros(n+1);vs=np.zeros(n+1);
    us[0]=1; vs[0]=0;
    for k in range(n):
        u1=us[k] + 0.5*h*((b-2*a)*us[k]+2*(b-a)*vs[k])
        v1=vs[k] + 0.5*h*((a-b)*us[k]+(a-2*b)*vs[k])
        us[k+1]=us[k]+h*((b-2*a)*u1+2*(b-a)*v1)
        vs[k+1]=vs[k]+h*((a-b)*u1+(a-2*b)*v1)
    return ts,us,vs,h

def imp_euler(h,tmin=0.0,tmax=1.0,u0=1,v0=0):
    """Numerical approximation to the stiff ODE in part 2 of the assignment brief of u and v over time t by the implicit Euler method
    args:
        h: step-size in time domain
        tmin: the smallest time value in your time-domain.
        tmax: the largest time value in your time domain.
        u0: initial condition on u
        v0: initial condition on v
    returns:
        ts: array containing n+1 time values 
        us: array containing n+1 midpoint approximations to u(t)
        vs: array containing n+1 midpoint approximations to v(t)
    """
    n=int((tmax-tmin)/h) #calculates n from h value given, parsed to an integer
    A=np.matrix([[1+(2*a-b)*h, 2*(a-b)*h],[(b-a)*h, 1+(2*b-a)*h]]) #Array A detailed and derived in the report 
    ts=np.linspace(tmin,tmax,n+1)
    us=np.zeros(n+1);vs=np.zeros(n+1)
    us[0]=u0;vs[0]=v0
    for k in range(n):
        Ainverse=np.linalg.inv(A) #takes inverse of A
        xk=np.matrix([[us[k]],
                      [vs[k]]]) #column vector of the kth u and v values 
        x1=Ainverse*xk #matrix multiplication to give the k+1th u and v values
        us[k+1]=x1[0] # appends the k+1th u value to the u array
        vs[k+1]=x1[1] # appends the k+1th v value to the v array
    return ts,us,vs
        
## Part 1b)
#Plot a time- domain graph and a phase plot for the time-domain t = [0,4]. 
fig1=plt.figure()
ts,us,vs,h= midpoint(0.0,4.0,101)
plt.plot(ts,us,'-',label="approx.")
u_exact=[]
for i in ts:
    u_exact.append(np.exp(-i)*(np.cos(2*i)+0.5*np.sin(2*i))) #analytical solution for u at time t
plt.plot(ts,u_exact,'o',label="exact",markersize=2.5) #plot analytical solution for u
plt.legend()
plt.xlabel("$t$",fontsize=16)
plt.ylabel("$u(t)$",fontsize=16)
fig1.savefig('1a_time-dom.jpg')

fig2=plt.figure()
plt.plot(us,vs,'-') #plot us against vs, midpoint method values
plt.xlabel("$u(t)$",fontsize=16)
plt.ylabel("$v(t)$",fontsize=16)
fig2.savefig('2a_phaseplot.jpg')


##part 1c) 

fig3=plt.figure()
hs,es=error_func(0.0,4.0,nb=0,nt=21,nstep=1,t=2) #run error function for given parameters
plt.plot(hs,es,'o',label='$error$')
#stepvalsq=[]
plt.plot(hs,[i**2 for i in hs],'-',label='$h^2$') #plot guideline for (h,h^2) points 
plt.xscale('log')
plt.yscale('log')
plt.xlabel("h",fontsize=16)
plt.ylabel("$\epsilon$",fontsize=16)
plt.legend()
fig3.savefig('error.jpg')


##Part 2b)
fig4=plt.figure();fig5=plt.figure()
tlin=np.linspace(0.0,1.0,100) #time values
u2_exact=[2*np.exp(-k)-np.exp(-200*k) for k in tlin] #analytical solution for u
v2_exact=[-np.exp(-k)+np.exp(-200*k) for k in tlin] #analytical solution for v
nary=(400,200,100,50) #array of n values that give required h values in function
for i in range(len(nary)):
    ts2,us2,vs2,h2=midpoint2(0.0,1.0,nary[i],1,200)  #midpoint method approximation for ith index of nary
    
    #generates subplot to add plots of different h values to. These are plots of time against u.
    ax1=fig4.add_subplot(2,2,i+1,title="h="+str(1/nary[i])) 
    ax1.plot(tlin,u2_exact,'-',linewidth=1,label="exact") #analytical solution plot
    ax1.plot(ts2,us2,'o',markersize=1,label="approx") #midpoint method plot
    ax1.set_xlabel("$t$",fontsize=16)
    ax1.set_ylabel("$u(t)$",fontsize=16)
    
    #generates subplot to add plots of different h values to. These are plots of time against v.
    ax2=fig5.add_subplot(2,2,i+1,title="h="+str(1/nary[i]))
    ax2.plot(ts2,vs2,'o',markersize=1,label="approx") #midpoint method plot
    ax2.plot(tlin,v2_exact,'-',linewidth=1,label="exact")#analytical solution plot
    ax2.set_xlabel("$t$",fontsize=16)
    ax2.set_ylabel("$v(t)$",fontsize=16)

ax1.legend(loc=0); ax2.legend(loc=0)
fig4.tight_layout();fig5.tight_layout() #spaces the subplots so that all titles and scales can be seen clearly and the plots don't overlap
fig4.savefig('t_ut_sub.jpg');fig5.savefig('t_vt_sub.jpg')


##Part 2c) 
#Implements the backwards Euler method for the h values used above and plots 8 subplots. 4 plotting t against u and 4 plotting t against tv for different h values
a=1;b=200; #some initial a and b values for our stiff ODE
fig6=plt.figure();fig7=plt.figure()
hary=(0.0025,0.005,0.01,0.02) #array of h values
for i in range(len(hary)):
    ts3,us3,vs3=imp_euler(hary[i]) #carries out implicit Euler method
    ax3=fig6.add_subplot(2,2,i+1,title="h="+str(hary[i]))
    ax3.plot(tlin,u2_exact,'-',linewidth=1,label="exact") #analytic solution of t against u
    ax3.plot(ts3,us3,'o',markersize=1,label="approx") #implicit euler method approximation of t against u
    ax3.set_xlabel("$t$",fontsize=16)
    ax3.set_ylabel("$u(t)$",fontsize=16)
    
    ax4=fig7.add_subplot(2,2,i+1,title="h="+str(hary[i]))
    ax4.plot(ts3,vs3,'o',markersize=1,label="approx") #implicit euler method approximation of t against v
    ax4.plot(tlin,v2_exact,'-',linewidth=1,label="exact") #analytic solution of t against v
    ax4.set_xlabel("$t$",fontsize=16)
    ax4.set_ylabel("$v(t)$",fontsize=16)

ax3.legend(loc=0); ax4.legend(loc=0)
fig6.tight_layout();fig7.tight_layout()
fig6.savefig('back_euler_u.jpg');fig7.savefig('back_euler_v.jpg')
plt.show()