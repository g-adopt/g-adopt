def moistureContent( modelParameters, h, x, time):

    import firedrake as fd

    if modelParameters["modelType"] == "Haverkamp":

        alpha  = modelParameters["alpha"]
        beta   = modelParameters["beta"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        theta = thetaR + alpha*(thetaS - thetaR) / (alpha + abs(100*h)**beta) ;
    
    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]
        m = 1 - 1/n

        theta = ( thetaR + (thetaS - thetaR) / ( (1 + abs(alpha*h)**n)**m ) )

    elif modelParameters["modelType"] == "exponential":

        alphaT = modelParameters["alphaT"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        #theta = thetaR + (thetaS - thetaR) * fd.exp(h * alphaT) 
        a = 0.9251; b = 2.4041; c = 1.2302;
        theta = thetaR + (thetaS - thetaR) / (a + b*((-h)**c))

    else:

        print("Model type not recognised")

    #return fd.conditional(h <= 0, theta, thetaS)
    return theta

def relativePermeability( modelParameters, h, x, time):

    import firedrake as fd
    Ks    = modelParameters["Ks"]
    dx    = modelParameters["gridSpacing"]
  
    if modelParameters["modelType"] == "Haverkamp":

        A     = modelParameters["A"]
        gamma = modelParameters["gamma"]

        K =  Ks*( dx*0e-00 + A / ( A + pow(abs(100*h), gamma) ) )

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        m = 1 - 1/n

        K = Ks * ( 0.25*dx*5e-05 +  ( 1 - (alpha*abs(h))**(n-1) * (1 + (alpha*abs(h))**n)**(-m))**2 / ( 1 + (alpha*abs(h)**n )**(m/2)));


    elif modelParameters["modelType"] == "exponential":

        alphaK  = modelParameters["alphaK"]

       # K = Ks*fd.exp(h*alphaK)
        a = 0.9010; b = 10.0721;
        K = Ks *(1e-06 +  a * fd.exp(h * b))

    #return fd.conditional(h <= 0, K, Ks)
    return K

def waterRetention( modelParameters, h, x, time):

    import firedrake as fd
  
    if modelParameters["modelType"] == "Haverkamp":

        alpha  = modelParameters["alpha"]
        beta   = modelParameters["beta"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = ( -fd.sign(h)*alpha*beta*(thetaS - thetaR)*pow(abs(h),beta-1) ) / ( pow(alpha + pow(abs(h),beta),2) )

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]
        m = 1 - 1/n

        C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) ) 

    elif modelParameters["modelType"] == "exponential":

        alphaT  = modelParameters["alphaT"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = (thetaS - thetaR) * fd.exp(h * alphaT) * alphaT

    #return fd.conditional(h <= 0, C, 0)
    return C