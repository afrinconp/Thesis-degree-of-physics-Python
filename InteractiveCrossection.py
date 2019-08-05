import sys
from tkinter import *
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.special import eval_legendre 
from scipy.special import roots_legendre  
from scipy.special import erf  
from scipy.special import gamma, factorial
import numpy as np
import pandas as pd


#-----------------------------------------Parameters---------------------------------------#

e2= 1.44
N=150
l=100



#-------------------------------Legendre roots and function--------------------------------#

def root(x):  #zeros legendre
    zero=roots_legendre(x)[0]
    xi=(zero+1)/2
    return xi

def L(N,x): #eval legendre function
    b=eval_legendre (N,x)
    return b

def Legendre_functions(N,a): #Regulazed Legendre functions evaluated in a
    M=[]
    zerosN=root(N)
    k=0
    for xi in zerosN:
        p=(-1)**(N+k+1)*(1/xi)*(a*xi*(1.-xi))**(1./2)*1/(a-a*xi)
        k=k+1
        M.append(p)
    MM=np.array(M)
    return MM




#----------------------------------------Potential--------------------------------------------#

def V(x,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
    aa=Vn(x,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)
    bb=Vc(x,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)
    cc=Vl(x,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)
    return aa+bb+cc

def Vl(x,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
    cent=h2mu*l*(l+1)/(x*x)
    return cent

def Vn(x,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
	xi=(x-Rr*At**(1/3))/Ar
	ff=(1+np.exp(xi))**(-1)
	VN=-V0*ff
	return VN

def Vc(x,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
	if x>R0:
		a=zp*zt*e2/x
	else:
		a=zp*zt*(e2/(2*R0))*(3-np.power(x/R0,2))
	return a

   
def Up(x,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
    R=V(x,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)+1j*W1(x,Ri,Ai,Wv,At)
    return R

def f1(x,Ri,Ai,At):
	xi=(x-Ri*At**(1/3))/Ai
	ff=(1+np.exp(xi))**(-1)
	return ff

def derf1(x,Ri,A):
	dx=1e-9
	df=(f1(x+dx/2,Ri,A)-f1(x-dx/2,Ri,A))/dx
	return df
	

def W1(x,Ri,Ai,Wv,At):
	WEr=Wv*f1(x,Ri,Ai,At)#-4*Wd*derf1(x,Ri,Ai)
	return -WEr


#--------------------------------------Matrix-------------------------------------------------#


def Cmatrix(zeros1,zeros2,a,N,l,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0): #Matrix Elements
	A = np.matrix(np.zeros((N, N), dtype = np.complex))
	i=0
	for xi in zeros2:
		j=0
		for xj in zeros1:
			if xi!=xj:
				element=((-1)**(i+j)/(a**2*(xi*xj*(1.-xi)*(1.-xj))**(0.5)))*(N**2+N+1+(xi+xj-2*xi*xj)/(xi-xj)**2-1./(1.-xi)-1./(1.-xj))
				A[i,j]=element*h2mu
			elif xi==xj:
				A[i,j]=((4*N**2+4*N+3)*xi*(1-xi)-6*xi+1)/(3*a**2*xi**2*(1-xi)**2)*h2mu+Up(a*xi,l,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)-E
			j=j+1
		i=i+1
	return A

#--------------------------------------R-Matrix-------------------------------------------------#
def RE(l,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a):
	zeros1=root(N)
	zeros2=root(N)
	Regulazed_Legendre_functions=Legendre_functions(N,a)
	C=Cmatrix(zeros1,zeros2,a,N,l,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)                  #C matrix
	Cinv = np.linalg.inv(C)                           #C inverse matrix
	RE1=np.dot(Cinv,Regulazed_Legendre_functions)        #Dot matrix vector
	p=RE1.tolist()[0]
	RE2=np.array(p)
	REa=(h2mu/a)*Regulazed_Legendre_functions.dot(RE2)          #R-matrix
	return REa
	
#------------------------------------Scattering matrix------------------------------------------#
def F(l,eta,x):                                   #F Coulumb function
    Fc=mp.coulombf(l,eta,x)
    return Fc

def G(l,eta,x):                                   #G Coulumb function
    Gc=mp.coulombg(l,eta,x)
    return Gc

def derivateH(m,l,eta,x):                         #Derivate of a function
    dx=1e-10
    h=(H(m,l,eta,x+dx/2)-H(m,l,eta,x-dx/2))/dx
    return h

def H(m,l,eta,x):                                 #Coulumb Hankel (+,-) function
    if m==1:
        h=G(l,eta,x)+1j*F(l,eta,x)
    else:
        h=G(l,eta,x)-1j*F(l,eta,x)
    return h
	
def U(l,eta,x,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a):                                 #Scattering matrix
    ul=(H(0,l,eta,x)-x*(RE(l,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a))*derivateH(0,l,eta,x))/(H(1,l,eta,x)-x*(RE(l,E,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a))*derivateH(1,l,eta,x))
    return ul 


def ScatterMatrix(l,i,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a):
	k=(i/h2mu)**(1./2)
	eta=zp*zt*e2/(2*k*h2mu)
	scatterM=U(l,eta,k*a,i,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a)
	return scatterM

#------------------------------------------Sigma-----------------------------------------------#	
def sigma0(eta):
	h=gamma(1+1j*eta)
	tt=np.angle(h)
	return tt

def sigmal(L,eta):
	h=gamma(1+L+1j*eta)
	tt=np.angle(h)
	return tt

#-------------------------------------Elastic Cross Section-------------------------------------#

def f(i,l,x,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a):
	k=(i/h2mu)**(1./2)
	eta=zp*zt*e2/(2*k*h2mu)
	M=[]
	fthetha2=-(eta/(2*k*np.sin(x/2)*np.sin(x/2)))*np.exp(2j*sigma0(eta))*(np.exp(-1j*eta*np.log((np.sin(x/2)*np.sin(x/2)))))
	L=0
	while L<=l:
		print(L)
		b=eval_legendre(L,np.cos(x))
		b=np.array(b)
		MatrixS=ScatterMatrix(L,i,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a)
		fthetha1=complex((1/(2*1j*k))*(2*L+1)*np.exp(2*1j*sigmal(L,eta))*(MatrixS-1)) 
		fthetha1=fthetha1*b
		ffthetha1=np.array(fthetha1)
		M.append(ffthetha1)
		L=L+1
	Ma=M[0]
	c=0
	for i in range(len(M)-1):               #Plus componentes of Wa
		c=c+1
		Ma=Ma+M[c]
	return (Ma,fthetha2)

#---------------------------------------------Plot-----------------------------------------------------------#

def salida(Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a,Energy):
	grados=np.arange(0.1,14,0.001)
	radianes=np.radians(grados)
	FTH=f(Energy,l,radianes,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a)
	FTHa,FTHb=FTH[0],FTH[1]
	ElasticCrosssection=(np.abs((FTHa+FTHb)/FTHb)**2)
	plt.semilogy(grados,ElasticCrosssection,label="V={}".format(V0))
	plt.title("Elastic Cross Section")
	plt.ylabel("σ/σR")
	plt.xlabel("θ(deg)")
	plt.legend()
	plt.savefig('foo.png')
	plt.show()

def salida2(Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0):
	radio=np.arange(0.01,33,0.01)
	DEFI=[]
	for i in range(5):
		pop=[]
		for j in radio:
			Points=V(j,i,Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)
			pop.append(Points)
		DEFI.append(pop)
	plt.title("Real Potential")
	plt.ylabel("V(r)")
	plt.xlabel("r")
	top=30
	bottom=min(DEFI[0])
	for graph in range(5):
		plt.ylim(bottom, top)  
		plt.title("Real Potential")
		plt.plot(radio,DEFI[graph],label="l={}".format(graph))
	plt.legend()
	plt.show()
			
		
###key down function
def click():
	Ap=textentry1.get() #this will collect the text from the text entry box
	At=textentry2.get() #this will collect the text from the text entry box
	zp=textentry3.get()
	zt=textentry4.get()
	Ap,At,zp,zt=float(Ap),float(At),float(zp),float(zt)
	h2mu=20.736*(Ap+At)/(Ap*At)
	Rr=textentry5.get() #this will collect the text from the text entry box
	Ar=textentry6.get() #this will collect the text from the text entry box
	Ri=textentry7.get()
	Ai=textentry8.get()
	Wv=textentry9.get() #this will collect the text from the text entry box
	V0=textentry10.get() #this will collect the text from the text entry box
	rc=textentry11.get() #this will collect the text from the text entry box
	a=textentry12.get() #this will collect the text from the text entry box
	Energy=textentry13.get()
	Rr,Ar,Ri,Ai,Wv,V0,rc,a,Energy=float(Rr),float(Ar),float(Ri),float(Ai),float(Wv),float(V0),float(rc),float(a),float(Energy)
	R0=rc*At**(1./3)
	salida(Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0,a,Energy)

def POTENTIAL():
	Ap=textentry1.get() #this will collect the text from the text entry box
	At=textentry2.get() #this will collect the text from the text entry box
	zp=textentry3.get()
	zt=textentry4.get()
	Ap,At,zp,zt=float(Ap),float(At),float(zp),float(zt)
	h2mu=20.736*(Ap+At)/(Ap*At)
	Rr=textentry5.get() #this will collect the text from the text entry box
	Ar=textentry6.get() #this will collect the text from the text entry box
	Ri=textentry7.get()
	Ai=textentry8.get()
	Wv=textentry9.get() #this will collect the text from the text entry box
	V0=textentry10.get() #this will collect the text from the text entry box
	rc=textentry11.get() #this will collect the text from the text entry box
	Rr,Ar,Ri,Ai,Wv,V0,rc=float(Rr),float(Ar),float(Ri),float(Ai),float(Wv),float(V0),float(rc)
	R0=rc*At**(1./3)
	salida2(Ap,At,zp,zt,h2mu,Rr,Ar,Ri,Ai,Wv,V0,rc,R0)

#### main:

window=Tk()
window.title("Elastic Cross Section ")
window.configure(background="black")

### My Photo
photo1=PhotoImage(file="Quatum.png")
photo2=PhotoImage(file="potential.png")
Label(window,image=photo1,bg="black").grid(row=0, column=1, sticky=W)
Label(window,image=photo2,bg="black").grid(row=0, column=3, sticky=W)

#### create label
Label(window, text="Ap", bg="black", fg="white", font="none 12 bold").grid(row=1, column=0, sticky=W)
Label(window, text="At", bg="black", fg="white", font="none 12 bold").grid(row=2, column=0, sticky=W)
Label(window, text="zp", bg="black", fg="white", font="none 12 bold").grid(row=3, column=0, sticky=W)
Label(window, text="zt", bg="black", fg="white", font="none 12 bold").grid(row=4, column=0, sticky=W)

Label(window, text="Rr", bg="black", fg="white", font="none 12 bold").grid(row=6, column=0, sticky=W)
Label(window, text="Ar", bg="black", fg="white", font="none 12 bold").grid(row=7, column=0, sticky=W)
Label(window, text="Ri", bg="black", fg="white", font="none 12 bold").grid(row=8, column=0, sticky=W)
Label(window, text="Ai", bg="black", fg="white", font="none 12 bold").grid(row=9, column=0, sticky=W)
Label(window, text="Wv", bg="black", fg="white", font="none 12 bold").grid(row=6, column=2, sticky=W)
Label(window, text="V0", bg="black", fg="white", font="none 12 bold").grid(row=7, column=2, sticky=W)
Label(window, text="rc", bg="black", fg="white", font="none 12 bold").grid(row=8, column=2, sticky=W)
Label(window, text="zero Potential", bg="black", fg="white", font="none 12 bold").grid(row=9, column=2, sticky=W)
Label(window, text="Energy", bg="black", fg="white", font="none 12 bold").grid(row=5, column=2, sticky=W)


#### create a text entry box

	
textentry1=Entry(window, width=7, bg="white")
textentry1.grid(row=1, column=1, sticky=W)        #Ap
textentry2=Entry(window, width=7, bg="white")
textentry2.grid(row=2, column=1, sticky=W)        #At
textentry3=Entry(window, width=7, bg="white")
textentry3.grid(row=3, column=1, sticky=W)        #zp
textentry4=Entry(window, width=7, bg="white")
textentry4.grid(row=4, column=1, sticky=W)        #zt
textentry5=Entry(window, width=7, bg="white")
textentry5.grid(row=6, column=1, sticky=W)        #Rr
textentry6=Entry(window, width=7, bg="white")
textentry6.grid(row=7, column=1, sticky=W)        #Ar
textentry7=Entry(window, width=7, bg="white")
textentry7.grid(row=8, column=1, sticky=W)        #Ri
textentry8=Entry(window, width=7, bg="white")
textentry8.grid(row=9, column=1, sticky=W)        #Ai
textentry9=Entry(window, width=7, bg="white")
textentry9.grid(row=6, column=3, sticky=W)        #Wv
textentry10=Entry(window, width=7, bg="white")
textentry10.grid(row=7, column=3, sticky=W)       #V0
textentry11=Entry(window, width=7, bg="white")
textentry11.grid(row=8, column=3, sticky=W)       #rc
textentry12=Entry(window, width=7, bg="white")
textentry12.grid(row=9, column=3, sticky=W)       #Zero potential
textentry13=Entry(window, width=7, bg="white")
textentry13.grid(row=5, column=3, sticky=W)       #Energy




##add a submit button
Button(window, text="SUBMIT",width=12,command=click).grid(row=10, column=3, sticky=W)
Button(window, text="POTENTIAL",width=12,command=POTENTIAL).grid(row=10,column=1,sticky=W)


window.mainloop()

