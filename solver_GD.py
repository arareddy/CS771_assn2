import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math
from scipy.sparse.linalg import norm

def hinge_loss(yhat,y):
    return max(0, 1 - y*yhat)

def main():
    # Get training file name from the command line
    traindatafile = sys.argv[1];
    # For how many iterations do we wish to execute GD?
    n_iter = int(sys.argv[2]);
    # After how many iterations do we want to timestamp?
    spacing = int(sys.argv[3]);
    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);

    Xtr = tr_data[0]; # Training features in sparse format
    Ytr = tr_data[1]; # Training labels

	# We have n data points each in d-dimensions
    n, d = Xtr.get_shape();

	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
    Ytr = 2*(Ytr - 1.5);
    Ytr = Ytr.astype(int);

    Xval = Xtr[240000:] #validation set, take 20% of the initial data
    Yval = Ytr[240000:]
    Yval = csr_matrix(Yval) #to use some functions later

    Xtr = Xtr[:240000] #actual training set
    Ytr = Ytr[:240000]
    Ytr = csr_matrix(Ytr) #to use some functions later

	# Optional: densify the features matrix.
	# Warning: will slow down computations
#	Xtr = Xtr.toarray();

	# Initialize model
	# For primal GD, you only need to maintain w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
    w = csr_matrix((1, d));
    wbar = w;

	# We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(math.ceil(n_iter/spacing));
    tick_vals = np.zeros(math.ceil(n_iter/spacing));
    obj_val = np.zeros(math.ceil(n_iter/spacing));

    tick = 0;

    ttot = 0.0;
    t_start = datetime.now();

    for t in range(n_iter):
		### Doing primal GD ###

        WXdot = csr_matrix.dot(w,Xtr.T);
        NegSign = (Ytr.multiply(WXdot) - np.ones(Xtr.shape[0]))< 0;
        NegSign = csr_matrix(NegSign);
        hinge_loss_derivative = -(csr_matrix.dot(NegSign.multiply(Ytr),Xtr));

		# Compute gradient
        g = w + hinge_loss_derivative;
        #g = g.toarray();
        #g.reshape(1,d); # Reshaping since model is a row vector

		# Calculate step length. Step length may depend on n and t
	#	eta = h(n) * 1/math.sqrt(t) * ...;
        eta = (30/n) * 1/math.sqrt(t+1);

		# Update the model
        w = w - eta * g;

		# Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
        wbar = (t*wbar + w)/(t+1);
        #print(t);

		# Take a snapshot after every few iterations
		# Take snapshots after every spacing = 5 or 10 GD iterations since they are slow
        if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
            t_now = datetime.now();
            delta = t_now - t_start;
            time_elapsed[tick] = ttot + delta.total_seconds();
            ttot = time_elapsed[tick];
            tick_vals[tick] = tick;
            #print("Mamma mia");
            #print((NegSign.T).shape);
            #print(Ytr.multiply(WXdot).shape);
            #print(np.ones(Xtr.shape[0]).shape);
            #hinge_loss_sum = (csr_matrix.dot(NegSign.T, np.ones(Xtr.shape[0]) - Ytr.multiply(WXdot))).sum();

            WXdot = csr_matrix.dot(w,Xval.T);
            NegSign = (Yval.multiply(WXdot) - np.ones(Xval.shape[0]))< 0;
            NegSign = csr_matrix(NegSign);

            #WXdot = csr_matrix.dot(wbar,Xval.T);
            #NegSign = (Yval.multiply(WXdot) - np.ones(Xval.shape[0]))< 0;
            #NegSign = csr_matrix(NegSign);

            #WXdot = csr_matrix.dot(wbar,Xtr.T);
            #NegSign = (Ytr.multiply(WXdot) - np.ones(Xtr.shape[0]))< 0;
            #NegSign = csr_matrix(NegSign);

            #hinge_loss_sum = (NegSign.multiply(np.ones(Xtr.shape[0]) - Ytr.multiply(WXdot))).sum();
            hinge_loss_sum = (NegSign.multiply(np.ones(Xval.shape[0]) - Yval.multiply(WXdot))).sum();
            #print("oo la la");
            #obj_val[tick] = 0.5*(w.dot(w.T)).toarray() + hinge_loss_sum; # Calculate the objective value f(w) for the current model w^t or the current averaged model \bar{w}^t
            obj_val[tick] = 0.5*(wbar.dot(wbar.T)).toarray() + hinge_loss_sum;
            #print(obj_val[tick]);
            print(hinge_loss_sum);
            tick = tick+1;
			# Start the timer again - training time!
            t_start = datetime.now();

	# Choose one of the two based on whichever works better for you
    w_final = w.toarray();
	# w_final = wbar.toarray();
    np.save("model_GD.npy", w_final);

if __name__ == '__main__':
    main()

"""
B = []
			temp = []
			B = w.dot(small_XTR.T)
			temp = (small_YTR.multiply(B) - np.ones(small_XTR.shape[0]))< 0
			hinge_dash = 0
			temp = csr_matrix(temp)
			hinge_dash = (temp.multiply(small_YTR)).dot(small_XTR)

			# Compute gradient
			g = w - hinge_dash;
			g = g.toarray()
			g.reshape(1,d); # Reshaping since model is a row vector

			# Calculate step lenght. Step length may depend on n and t
			eta = C * 1/(math.sqrt(t+1));
			w = w - eta * g;


			count = count +1

			hinge = 0
			hinge = temp.multiply(np.ones(small_XTR.shape[0]) - small_YTR.multiply(B))
"""
