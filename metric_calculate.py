import torch
import numpy as np
import random
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm
from scipy.integrate import solve_ivp
import time

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of Attention Maps')
    parser.add_argument('-np',
                        '--npy_save_path',
                        default='./data/Attention_maps/train',
                        type=str,
                        required=False,
                        dest="npy_save_path",
                        help='npy save path')
    parser.add_argument('-m',
                        '--metric_save_path',
                        default='./data/Metrics/train',
                        type=str,
                        required=False,
                        dest="metric_save_path",
                        help='metric save path')
    
    args = parser.parse_args()
    return args

# set the random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.Generator().manual_seed(int(seed))

class AttentionMetrics:
    def __init__(self, attention_maps=None):
        """
        Initialize the AttentionMetrics class, which receives attention_maps data.
        attention_maps: A list of lists, with a shape of [T][L], where each L is a 16x16 2D array.
        """
        self.attention_maps = np.array(attention_maps)
        self.T, self.L, self.H, self.W = self.attention_maps.shape  # obtain the shape of attention_maps
        self.time_cost = 0  # time cost for computing metrics
        self.T = 50

    def attention_change_rate(self):
        """
        Calculate the attention distribution change rate for each time step (excluding the first and last token).
        Returns: The average change rate for each time step, with shape (T-1,)
        """
        delta_A = np.zeros((self.T-1, self.L))  # Store the change rate for each time step
        for t in range(1, self.T):
            for l in range(1, self.L-1):
                delta_A[t-1, l] = np.linalg.norm(self.attention_maps[t][l] - self.attention_maps[t-1][l])  
        # Calculate average change rate
        delta_A_mean = np.mean(delta_A, axis=(1)) 
        return delta_A_mean

    def attention_change_rate_eos(self):
        """
        Calculate the attention distribution change rate for each time step for the <EOS> token.
        Returns: The average change rate for each time step, with shape (T-1,)
        """
        # Initialize an array to store the change rate for each time step
        delta_A = np.zeros((self.T-1))  
        for t in range(1, self.T):  
            l = self.L - 1  # Set l to the last layer index
            delta_A[t-1] = np.linalg.norm(self.attention_maps[t][l] - self.attention_maps[t-1][l])  
        return delta_A
    
    def compute_similarity(self, M_t):
        # M_t: [L-1, 16, 16], Note that M_t is a tensor (or matrix) representing attention maps.
        L = M_t.shape[0]
        similarity_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i != j:
                    similarity_matrix[i, j] = np.linalg.norm(M_t[i] - M_t[j])  # frobenius Norm
                else:
                    similarity_matrix[i, j] = 0 
                    
        sim_max = np.max(similarity_matrix)
        sim_min = np.min(similarity_matrix)
        diff = sim_max + sim_min
        
        frobenius_norms_final = (diff - similarity_matrix) / diff 
        
        return frobenius_norms_final

    def compute_laplacian(self, W):
        # W: [L, L] similarity matrix
        n = W.shape[0] 
        A = np.zeros_like(W) 
        
        # 1) Fill in the non-diagonal elements of A: A[i,j] = W[j,i]
        #    This step transposes the off-diagonal elements of W and assigns them to A
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = W[j, i]
        
        # 2) Fill in the diagonal elements of A: A[i,i] = - ∑_{k≠i} W[k,i]
        #    This step calculates the negative sum of the i-th column of W, excluding the diagonal element W[i,i]
        #    It is equivalent to - ( sum(W[:, i]) - W[i,i] )
        #    where sum(W[:, i]) gives the sum of the i-th column and W[i,i] is the diagonal element to be excluded
        for i in range(n):
            A[i, i] = - (np.sum(W[:, i]) - W[i, i])
        
        return A
        
    def system_dynamics(self, t, X, F, A, c):
        '''
        Define the system dynamics function
        '''
        # Convert the time variable to an integer
        t = int(t)
        # X is the state vector (length L)
        A_t = A[t]
        return np.dot(F, X) + c * np.dot(A_t, X)

    def node_trace(self, c=1):
        '''
        Complex dynamics process
        '''
        begin_time = time.time() 
        # Node stability
        L = self.L - 1  # Number of nodes, excluding the BOS token
        F = np.diag(np.ones(L) * (-1))  # Assume the system decay rate is -1 for all nodes
        F[-1][-1] = -10  # Set the decay rate of the last node to -10
        
        X = []  
        A = [] 

        for t in range(0, self.T):
            # Obtain the attention map at time step t
            M_t = self.attention_maps[t][1:, :, :]  # [L-1, 16, 16], excluding the BOS token

            # Calculate the similarity matrix
            W = self.compute_similarity(M_t)  # [L-1, L-1]

            # Calculate the Laplacian matrix
            A_t = self.compute_laplacian(W)  # [L-1, L-1]

            # Calculate the derivative of the state equation
            # X(t) represents the norm of the attention map at each node
            X_t = np.linalg.norm(M_t, axis=(1, 2))  # [L-1]
            
            X.append(X_t)
            A.append(A_t)

        # Initial conditions
        X0 = X[0]

        # Time span for the simulation
        t_span = (0, self.T-1) # 50 steps
        # t_eval = np.linspace(0, self.T-1, 1000)

        # Numerical solution of the system
        sol = solve_ivp(self.system_dynamics, t_span, X0, args=(F, A, c))

        X_avg = np.mean(sol.y[:-2, :], axis=0) 
        
        # print()
        
        # print(X[0][-1])
        
        # print(sol.y[-1, :][0])

        # RST
        differ = sol.y[-1, :] - X_avg
        
        differ_speed = []
        for i in range(1, sol.y.shape[1]):
            # for each time step
            delta_eos = sol.y[-1, i] - sol.y[-1, i-1] # change rate of the <EOS> token
            delta_others = []
            for j in range(sol.y.shape[0]-1):
                # for each node
                delta_others.append(sol.y[j, i] - sol.y[j, i-1])
            delta_others = np.array(delta_others) # [L-1]
            delta_avg = np.mean(delta_others) # average of the other nodes' change rate
            differ_speed.append(delta_eos - delta_avg)
        
        differ_speed = np.array(differ_speed)
        
        differ = differ_speed.tolist()[:100]
        
        end_time = time.time()
        
        duration = end_time - begin_time
        
        self.time_cost = duration
        
        return differ
    
    def save_metrics(self, filename="attention_metrics.npy"):
        """
        Save the results of change rate, entropy, concentration, and change acceleration to an npy file.
        filename: The name of the file to save, default is "attention_metrics.npy"
        """
        # Calculate all metrics
        begin_time = time.time() 
        delta_A_mean = self.attention_change_rate()
        delta_A_eos = self.attention_change_rate_eos()
        attention_node_trace = self.node_trace(c=1)
        end_time = time.time()
        duration = end_time - begin_time
        self.time_cost = duration

        # Store the results in a dictionary
        metrics = {
            'delta_A_mean': delta_A_mean,
            'delta_A_eos': delta_A_eos,
            'attention_node_trace': attention_node_trace,
        }

        # Save the dictionary to an npy file
        np.save(filename, metrics)

    def load_metrics(self, filename="attention_metrics.npy"):
        """
        Load the saved metrics from an npy file.
        filename: The name of the file to save, default is "attention_metrics.npy"
        """
        metrics = np.load(filename, allow_pickle=True).item()
        return metrics


def find_npy_file_path(npy_save_path):
    npy_file_path = []
    for backdoor_attack_method in os.listdir(npy_save_path):
        if os.path.isdir(os.path.join(npy_save_path, backdoor_attack_method)):
            for backdoor_model_name in os.listdir(os.path.join(npy_save_path, backdoor_attack_method)):
                if os.path.isdir(os.path.join(npy_save_path, backdoor_attack_method, backdoor_model_name)):
                    for npy_file in os.listdir(os.path.join(npy_save_path, backdoor_attack_method, backdoor_model_name)):
                        if npy_file.endswith(".npy"):
                            npy_file_path.append(os.path.join(npy_save_path, backdoor_attack_method, backdoor_model_name, npy_file))

    return npy_file_path

def main():
    set_seed(42)

    # define and parse arguments
    args = create_parser()

    npy_file_paths = find_npy_file_path(args.npy_save_path)
    
    print("processing {:.1f} attention maps...".format(len(npy_file_paths)))

    time_cost = []
    
    for i, npy_file_path in tqdm(enumerate(npy_file_paths)):
        attention_maps_numpy = np.load(npy_file_path)
        metrics = AttentionMetrics(attention_maps_numpy)                
        backdoor_id = npy_file_path.split('/')[-2]
        backdoor_method = npy_file_path.split('/')[-3]
        npy_id = int(npy_file_path.split('/')[-1].split('.')[0].split('_')[-1])

        metric_save_path = os.path.join(args.metric_save_path, backdoor_method, str(backdoor_id))
        if not os.path.exists(metric_save_path):
            os.makedirs(metric_save_path)
        metric_save_path = metric_save_path + f"/attention_metrics_{str(npy_id)}.npy"

        metrics.save_metrics(metric_save_path)

        time_cost.append(metrics.time_cost)
        i+=1
    
    if len(time_cost) > 0:
        time_avg = np.mean((np.array(time_cost)))
        time_var = np.var((np.array(time_cost)))
        
        print("time duration: ", time_avg)
        print("time variance: ", time_var)
    
    print("done!")
    
if __name__=="__main__":
    main()