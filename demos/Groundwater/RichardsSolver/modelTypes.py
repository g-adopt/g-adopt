import firedrake as fd


def moistureContent(modelParameters, h, x, time):

    thetaR = modelParameters["thetaR"]
    thetaS = modelParameters["thetaS"]

    if modelParameters["modelType"] == "Haverkamp":

        alpha = modelParameters["alpha"]
        beta = modelParameters["beta"]

        theta = thetaR + alpha*(thetaS - thetaR) / (alpha + abs(h)**beta)

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha = modelParameters["alpha"]
        n = modelParameters["n"]
        m = 1 - 1/n

        theta = (thetaR + (thetaS - thetaR) / ((1 + abs(alpha*h)**n)**m))

    elif modelParameters["modelType"] == "exponential":

        alpha = modelParameters["alpha"]
        theta = thetaR + (thetaS - thetaR) * fd.exp(h * alpha)

    else:

        print("Model type not recognised")

    return fd.conditional(h <= 0, theta, thetaS)


def relativePermeability(modelParameters, h, x, time):

    Ks = modelParameters["Ks"]

    if modelParameters["modelType"] == "Haverkamp":

        A = modelParameters["A"]
        gamma = modelParameters["gamma"]

        K = Ks*(A/(A + pow(abs(h), gamma)))

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha = modelParameters["alpha"]
        n = modelParameters["n"]
        m = 1 - 1/n

        K = Ks*((1 - (alpha*abs(h))**(n-1) * (1 + (alpha*abs(h))**n)**(-m))**2 / (1 + (alpha*abs(h)**n)**(m/2)))

    elif modelParameters["modelType"] == "exponential":

        alpha = modelParameters["alpha"]
        K = Ks*fd.exp(h*alpha)

    return fd.conditional(h <= 0, K, Ks)


def waterRetention(modelParameters, h, x, time):

    if modelParameters["modelType"] == "Haverkamp":

        alpha = modelParameters["alpha"]
        beta = modelParameters["beta"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = (-fd.sign(h)*alpha*beta*(thetaS - thetaR)*pow(abs(h),beta-1)) / (pow(alpha + pow(abs(h),beta),2))

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha = modelParameters["alpha"]
        n = modelParameters["n"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]
        m = 1 - 1/n

        C = -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * ((alpha**n * abs(h)**n + 1)**(-m-1))

    elif modelParameters["modelType"] == "exponential":

        alpha = modelParameters["alpha"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = (thetaS - thetaR) * fd.exp(h * alpha) * alpha

    return fd.conditional(h <= 0, C, 0)
