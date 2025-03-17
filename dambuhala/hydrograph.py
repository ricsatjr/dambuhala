import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize


class hydrograph_BetaDist:
    """
    Class for creating a hydrograph from estimates of total volume and peak discharge, 
    and from the shape of a beta pdf.

    see https://en.wikipedia.org/wiki/Beta_distribution
    """
    def __init__(self,Vw,Qp,dt,
                 minpeakloc=0.1,
                 maxpeakloc=0.5,
                 grad1thresh_ratio=0.01,):
        """
        Initiallize class with the hydrograph constraints

        Arguments
            Vw                   hydrograph total water volume, in m^3
            Qp                   hydrograph peak discharge, in m^3/s
            dt                   hydrograph time step, in hours
            minpeakloc           minimum x coordinate of beta pdf peak
            maxpeakloc           maximum x coordinate of beta pdf peak
            grad1thresh_ratio    maximum ratio of (gradient at x=1) to (gradient of pdf peak relative to x=1) 
        """
        self.Vw=Vw
        self.Qp=Qp
        self.dt=dt
        self.minpeakloc=minpeakloc
        self.maxpeakloc=maxpeakloc
        self.grad1thresh_ratio=grad1thresh_ratio
        self.generate_beta_params()
    
    def make_points(self,a,b,numpts=1000):
        """
        Generate beta pdf from x values

        Arguments
            a,b      parameters defining the beta distribution; a,b>0
            numpts   number of points to generate

        Return
            x
            pdf
        """
        x = np.linspace(0, 1, numpts)
        # Calculate the PDF values
        pdf = beta.pdf(x, a, b)
        return x,pdf
    
    def check_beta_params(self,a=2,b=5):
        """
        Check if parameters a,b produce a pdf that is shaped like a typical hydrograph.

        Constrainst for an acceptable beta pdf are:
        1. at x=0:
            gradient>0
        2. at peak:
            maxpeakloc>=x[peak]>=minpeakloc
        3. at x=1:
            curvature is negative (concave up)
            gentle gradient relative to the peak

        Sample acceptable pdf shape is when a=2 and b=5 (https://en.wikipedia.org/wiki/Beta_distribution#/media/File:Beta_distribution_pdf.svg)
       

        Args
            a,b                 parameters defining the beta distribution; a,b>0
        
        Return
            True                if all constraints are satisfied; False if otherwise
        
        """
        c=True
        while c==True:
            # Generate x and pdf values
            x,pdf = self.make_points(a, b)
            # Set constraints
            ## positive gradient at x=0
            c1=pdf[0]<pdf[1]
            c=c*c1
            ## minimum and maximum peak location
            c2a=x[np.argmax(pdf)]>=self.minpeakloc
            c2b=x[np.argmax(pdf)]<=self.maxpeakloc
            c=c*c2a*c2b
            ## negative gradient at x=1
            dx1=x[-1]-x[-2]
            dx2=x[-2]-x[-3]
            dx1p=x[-1]-x[np.argmax(pdf)]
            if dx1p==0:
                dx1p=self.dt*1e-6
            c3=(pdf[-1]-pdf[-2])/dx1<=0
            c=c*c3
            ## gentle gradient at x=1
            c4=abs((pdf[-1]-pdf[-2])/dx1) <= self.grad1thresh_ratio*abs((pdf[-1]-pdf[np.argmax(pdf)])/dx1p)
            c=c*c4
            ## concave up at x=1 
            c5=(pdf[-1]-pdf[-2])/dx1>=(pdf[-2]-pdf[-3])/dx2
            c=c*c5
            break
        
        return c

    def generate_beta_params(self,
                             min_a=1.5, max_a=5,
                             min_b=1,max_b=10,
                             n=100,randomseed=None,
                             plot=False):
        """
        Generate sets of beta parameters which would yield a beta pdf
        that is shaped like a flood hydrograph

        Arguments
            min_a,max_a         minimum and maximum values for the beta distribution parameter, a
            min_b,max_b         minimum and maximum values for the beta distribution parameter, b
            n                   number of parameter pairs (a,b) to generate and test
            randomseed          integer to set the random seed
            plot                bool, make a plot of valid beta distribution shapes if True
            
        """
        np.random.seed(randomseed)
        alist = np.round(np.random.uniform(low=min_a, high=max_a, size=n),2)
        blist = np.round(np.random.uniform(low=min_b, high=max_b, size=n),2)

        valid_beta=[]
        for i in range(len(alist)):
            if self.check_beta_params(alist[i],blist[i]):
                valid_beta.append((alist[i],blist[i]))
        if plot:
            plt.figure()
            for params in valid_beta:
                # Generate x and pdf values
                x,pdf = self.make_points(params[0], params[1])
                plt.gca().plot(x,pdf,alpha=0.2,c='k')
        self.valid_beta_params=valid_beta

    def compute_deltaVw(self,test_duration,a,b):
        """
        Compute the differences between the total volumes of a test hydrograph and the actual hydrograph.
        Test hydrograph is generated by scaling a beta distribution determined by (a,b) to
        the actual hydrograph's peak discharge and duration. 

        The volume difference serves as the objective function for the optimizating the hydrograph duration.
        
        Args:
            a,b             parameters defining the beta distribution; a,b>0
            test_duration   candidate value for hydrograph duration, in hours, used to scale the beta distribution
            self.Qp         hydrograph peak discharge, in m^3/s, used to scale the beta distribution
            self.Vw         hydrograph total water volume, in m^3
            
            dt   hydrograph time step, in hours
            
        Returns
            delta_Vw        volume difference between actual and test hydrographs, in m^3
        """

        # generate beta pdf given a,b
        yb=self.make_points(a,b)[1] 
        # scale beta pdf x-axis to the test_duration  
        t = np.linspace(0, test_duration*3600, len(yb)) # set x units to seconds
        # scale beta pdf  y-axis to the peak discahrge
        q = (yb/np.max(yb))*self.Qp     # set y units to m^3/s
        # Fit a spline to the data
        spline = UnivariateSpline(t, q, s=1)  # s=0 for an interpolating spline
        # get the integral to get area under the curve (convert hours to seconds for unit consistency)
        test_Vw=spline.integral(0,test_duration*3600)
        
        delta_Vw=abs(self.Vw-test_Vw)
        
        return delta_Vw

    def optimize_duration(self, a, b,initial_guess=24):
        
        """
        Optimizes the duration to match the total volume using minimize function.

        Requires an initialized hydrograph_BetaDist class.
    
        Args:
            a,b             parameters defining the beta distribution; a,b>0
            initial_guess:  initial guess for the duraiton, in hours
            self.Vw:        Total volume of the hydrograph (m3).
            
        Returns:
            self.T    the optimized duration in hours
        """

        # Define the objective function to be minimized
        def objective_function(duration):
            opdur=self.compute_deltaVw(duration,a,b)
            return opdur
        
        # Perform optimization, add bounds to ensure non-negative duration
        result = minimize(objective_function, 
                          initial_guess,
                          bounds=[(0, None)],                 # allow only duration >= 0
                          method='Nelder-Mead',
                          options={'fatol':0.00001*self.Vw})  # set tolerance relative to a fraction of the total volume

        
        T=result.x[0]
        self.T=T
        
        # Return the optimized hydrograph duration, in hours 
        return T

    def make_hydrograph(self,a,b,plot=False):
        """
        Generate a hydrograph represented by arrays of time, discharge, and volume
        for a given set of beta distribution parameters (a,b).

        Requires and initialized hydrograph_BetaDist class.

        Args:
            a,b            parameters defining the beta distribution
            self.dT        length of single timestep in the hydrograph, in hours.
            self.Qp        Peak discharge of the flood (m3/s).

        Returns:
            t  time array, in hours
            q  discharge array, in m^3/s
            v  volume array, in m^3
        """
        # get hydrograph duration (T)
        T=self.optimize_duration(a, b)
        # create array t from duration T and dt
        t=np.linspace(0,T,int(T/self.dt))
        # create required number of beta pdf points
        xb,yb=self.make_points(a,b,numpts=len(t))
        # compute discharge q at given t 
        q=beta.pdf(xb, a, b)*self.Qp/np.max(yb)
        # generate spline for the hydrograph
        spline = UnivariateSpline(t, q, s=1)  # s=0 for an interpolating spline
        # integrate spline per every dt to get volume per dt
        v=[]
        for i in range(len(t[:-1])):
           v.append(spline.integral(t[i],t[i+1]))
        
        self.t=t
        self.q=q
        self.v=v
        
        if plot:
            fig,ax=plt.subplots()
            plt.title(fr'$\beta: a={float(a):.2f}, b={float(b):.2f}$')
            plt.xlabel(r'$time\;(hours)$')
            
            ax=plt.gca()
            ax.bar(t[:-1],v,width=1.05*self.dt,align='edge',alpha=1,linewidth=0,edgecolor='none')
            ax.set_ylabel(r'$volume\;(m^3)$')
            ylim=ax.get_ylim()
            ax.set_ylim((0,ylim[1]))

            ax2=plt.gca().twinx()
            ax2.plot(t,q,c='k')
            ax2.set_ylabel(r'$discharge\;(m^3\;s^{-1})$')
            ylim=ax2.get_ylim()
            ax2.set_ylim((0,ylim[1]))
            ax2.set_xlim((0,self.T))

        return t,q,v

