import numpy as np
from numpy import linalg as LA
from mpi4py import MPI 
import networkx as nx 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class machine(object):
	def __init__(self,comm,size,rank,sample_num,di,graph):

		self.comm = comm
		self.size = size 
		self.id = rank 
		self.sample_num = sample_num
		self.di = di
		self.graph = graph

	def sample(self):

		gamma = np.random.choice([-1,1],(self.sample_num,1))
		feature = np.where(gamma > 0 ,np.random.normal(1,0.5,(self.sample_num,self.di)),\
			    np.random.normal(-1,0.5,(self.sample_num,self.di)))

		return gamma,feature

	def generate_agent_network(self):

		G = self.graph
		adj_mat = nx.adjacency_matrix(G).todense()	
		np.fill_diagonal(adj_mat,1)	
		weight_mat = np.array(adj_mat/np.sum(adj_mat,axis = 1).reshape(self.size,1))
		np.fill_diagonal(adj_mat,0)
		neighbor = adj_mat[self.id]

		return [neighbor,weight_mat]

class computation(object):

	def __init__(self,agent,iter_num,x0,step_size=0.05,epsilon=0.001):

		self.iter_num = iter_num
		self.agent = agent
		self.x0 = x0
		self.step_size = step_size
		self.epsilon = epsilon

	def com_log(self,x,t):

		s_l = self.agent.sample_num
		rho = 10
		gamma,sample = self.agent.sample()
		f = rho*LA.norm(x,axis = 1).reshape(t,1)/2
		e_x = gamma*sample.dot(x.T)	
		log_f = f + np.sum(np.log(1+np.exp(-1*e_x)),axis=0).reshape(t,1)/(len(sample))

		return log_f

	def communication_variable(self,x):

		now_agent,weight_mat = self.agent.generate_agent_network()

		for i in np.where(now_agent > 0)[1]:		
			self.agent.comm.send(x,dest = i)
		x = weight_mat[rank][rank]*x
		for i in np.where(now_agent > 0)[1]:
			comm_rec = self.agent.comm.recv(source = i)
			x += weight_mat[rank][i]*comm_rec

		return x

	def update_local(self,x0):

		t = 50

		sample_unit = np.random.normal(0,0.5,(t,self.agent.di))

		unit_vec = sample_unit/np.sqrt(np.sum(sample_unit**2,axis = 1)).reshape(t,1)
		eps_vec  = self.epsilon*unit_vec
		x1 = x0 + eps_vec
		x2 = x0 - eps_vec

		f_1 = self.com_log(x1,t)
		f_2 = self.com_log(x2,t)

		avg_gradient = np.sum(self.agent.di*(f_1 - f_2)*unit_vec/(2*self.epsilon),axis=0)/t
		x = x0
		x = x - self.step_size * avg_gradient

		x = self.communication_variable(x)

		return x

	def run(self):
		x = self.x0
		print(f"1\t{self.agent.id}\t{x}")
		for i in range(self.iter_num):
			x = self.update_local(x)
			if i+1 == self.iter_num:
				print(f"{self.iter_num}\t{self.agent.id}\t{x}")


def generate_graph(agent_num):
	while True:
		G = nx.binomial_graph(agent_num,p=0.5,seed=3)
		if nx.is_connected(G):
			break
	return G

agent_num = 4
sample_num = 10
di = 3
G = generate_graph(agent_num)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

agent = machine(comm,agent_num,rank,sample_num,di,G)
x0 = np.random.normal(100,2,(1,di))
logistic_f = computation(agent,10000,x0)
logistic_f.run()
