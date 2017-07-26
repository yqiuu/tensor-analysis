from copy import copy
import numpy as np, sympy as sym
try:
    from IPython.display import Latex, display
except ImportError:
    print "Can't import ipython notebook"

basis = None
metric = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, -1]])
	
def TensorIndex(rank):
	# Generate a full list of all possible indices
    # The algorithm is to turn a decimal number to a number system of dim
	dim = len(metric)
	inds_list = np.zeros([dim**rank, rank], dtype = int)
	k = np.arange(dim**rank, dtype = int)
	for j in xrange(rank):
		inds_list[:, rank - j - 1] = k%dim
		k /= dim
	return inds_list

def MetricTensor(ind_type = 1):
	if ind_type == 1:
		new_tensor = Tensor(2, 'g')
		new_tensor.ind_type = np.ones(2, dtype = int)  
		new_tensor.value = metric.astype(object)
		return new_tensor
	else:
		new_tensor = Tensor(2, 'g')
		new_tensor.ind_type = np.zeros(2, dtype = int)
		new_tensor.value = sym.Matrix(metric)**-1
		return new_tensor

def SetMetric(_basis, _metric):
	global basis, metric
	basis = np.array(_basis)
	metric = np.array(_metric)
	return MetricTensor()

def GenerateName(target, inds, disp_type, sgn = ''):
    str_list = ['{}^', '{}_']
    string = sgn + target.name
    for i in xrange(len(inds)):
        string += str_list[target.ind_type[i]]
        if disp_type == 0:
            string += ('{' + str(inds[i]) + '}')
        else:
            string += ('{' + sym.latex(basis[inds[i]]) + '}')
    string += '='
    return string    
                
def Disp(target, disp_type = 0, factor = True, collect = None):
    if type(target) == Tensor:
		target.simplify(factor = factor, collect = collect)
		inds_list = TensorIndex(target.rank)
		delete_list = []
		for i in xrange(len(inds_list)):
			if target.value[tuple(inds_list[i])] == 0:
				delete_list.append(i)
		inds_list = np.delete(inds_list, delete_list, axis = 0)
		if len(inds_list) == 0:
			print 'All components are zero'
		else:
			while len(inds_list) > 0:
				string = '\\begin{equation}'
				string += GenerateName(target, inds_list[0], disp_type)
				value = target.value[tuple(inds_list[0])]
				delete_list = [0]
				for i in xrange(1, len(inds_list)):
					inds = tuple(inds_list[i])
					if value == target.value[inds]:
						string += GenerateName(target, inds, disp_type)
						delete_list.append(i)
					elif value == -target.value[inds]:
						string += GenerateName(target, inds, disp_type, sgn='-')
						delete_list.append(i)
				string += sym.latex(value)
				string += '\\end{equation}'
				display(Latex(string))
				inds_list = np.delete(inds_list, delete_list, axis = 0)				
    else:
		target = sym.powsimp(target, deep = True)
		target = sym.cancel(target)
		target = sym.nsimplify(target, (sym.pi,))
		if factor == True:
			target = sym.factor(target)
		if collect != None:
			target = sym.collect(target, collect)
		display(Latex('\\begin{equation}' + sym.latex(target) + '\\end{equation}'))         
        
        
class Tensor(object):
	def __init__(self, rank, name = 'T'):
		'''index type = 0 vector/contravariant
		index type = 1 1-form/covariant'''
		dim = len(metric)
		rank = np.array(rank).flatten()
		if len(rank) == 1:
			self.rank = rank[0]
			self.ind_type = np.zeros(self.rank, dtype = int)
			self.is_tensor = True
		else:
			self.rank = len(rank)
			self.ind_type = np.array(rank).flatten()
			self.is_tensor = False 
		self.indices = np.zeros(self.rank, dtype = int)
		self.value = np.empty(self.rank*[dim], object)
		self.name = name
            
	def copy(self):
		new_tensor = Tensor(self.rank)
		new_tensor.is_tensor = self.is_tensor
		new_tensor.indices = np.copy(self.indices)
		new_tensor.ind_type = np.copy(self.ind_type)
		new_tensor.value = np.copy(self.value)
		new_tensor.name = self.name    
		return new_tensor        
            
	def simplify(self, factor = False, collect = None):
		inds_list = TensorIndex(self.rank)
		for ind in inds_list:
			self.value[tuple(ind)] = sym.powsimp(self.value[tuple(ind)], deep = True)
			self.value[tuple(ind)] = sym.cancel(self.value[tuple(ind)])
			self.value[tuple(ind)] = sym.nsimplify(self.value[tuple(ind)], (sym.pi,))
			if factor == True:
				self.value[tuple(ind)] = sym.factor(self.value[tuple(ind)])
			if collect != None:
				self.value[tuple(ind)] = sym.collect(self.value[tuple(ind)], collect)
		return self
		
	def contraction(self):
		dim = len(metric)
		new_rank = self.rank
		new_indices = self.indices
		new_value = self.value
		for inds in set(self.indices).intersection(set(-self.indices[self.indices < 0])):
			if new_rank > 2:
				new_rank -= 2
				i, j = np.where(new_indices == inds)[0][0], np.where(new_indices == -inds)[0][0]
				new_indices = np.delete(new_indices, (i, j))
				new_value = np.trace(new_value, axis1 = i, axis2 = j)
			else:
				new_value = np.trace(new_value)
				break
		else:
			# generate new tenosr
			new_type = (new_indices < 0).astype(int)
			new_tensor = Tensor(len(new_type), self.name)
			new_tensor.indices = new_indices 
			new_tensor.ind_type = new_type
			new_tensor.value = new_value			
			return new_tensor.simplify()
		return sym.simplify(new_value)		

	def __add__(self, other):
		dim = len(metric)
		if type(self) == type(other):
			if (self.rank == other.rank) and (set(self.indices) == set(other.indices)):
				inds_list_self = TensorIndex(self.rank)
				change_list = [0]*self.rank
				for i in xrange(self.rank):
					for j in xrange(self.rank):
						if other.indices[j] == self.indices[i]:
							change_list[j] = i
				inds_list_other = np.copy(inds_list_self)
				for i in xrange(self.rank):
					inds_list_other[:, i] = inds_list_self[:, change_list[i]]
				new_ind_value = np.empty(self.rank*[dim], object)
				for i in xrange(len(inds_list_self)):
					ind_self = tuple(inds_list_self[i])
					ind_other = tuple(inds_list_other[i])
					new_ind_value[ind_self] = self.value[ind_self] + other.value[ind_other]
				# generate new tenosr
				new_tensor = Tensor(self.rank)
				new_tensor.indices = self.indices
				new_tensor.ind_type = self.ind_type
				new_tensor.value = new_ind_value
				return new_tensor.simplify()
			else:
				raise ValueError('invalid addition')
		else:
			new_tensor = self.copy()
			new_tensor.value += other
			return new_tensor.simplify()

	def __radd__(self, other):
		return self.__add__(other)

	def __neg__(self):
		new_tensor = self.copy()
		new_tensor.value = -new_tensor.value
		return new_tensor

	def __sub__(self, other):
		return self.__add__(-other)
	
	def __rsub__(self, other):
		return (-self).__add__(other)

	def __mul__(self, other):
		dim = len(metric)
		if type(self) == type(other):
			if set(self.indices).intersection(set(other.indices)) == set():
				new_rank = self.rank + other.rank
				inds_list = TensorIndex(new_rank)
				new_value = np.empty(new_rank*[dim], object)
				for i in xrange(len(inds_list)):
					ind_new = tuple(inds_list[i])
					ind_self = tuple(inds_list[i][:self.rank])
					ind_other = tuple(inds_list[i][self.rank:])
					new_value[ind_new] = self.value[ind_self]*other.value[ind_other]
				# generate new tenosr
				new_indices = np.append(self.indices, other.indices)
				new_type = (new_indices < 0).astype(int)
				new_tensor = Tensor(len(new_type))
				new_tensor.indices = new_indices 
				new_tensor.ind_type = new_type
				new_tensor.value = new_value
				return new_tensor.contraction()
			else:
				raise IndexError("Indices are repeated")	 
		else:
			new_tensor = self.copy()
			new_tensor.value *= other
			return new_tensor.simplify()

	def __rmul__(self, other):
		return self.__mul__(other)
		
	def __div__(self, other):
		new_tensor = self.copy()
		new_tensor.value /= other
		return new_tensor
	
	def updown(self, i_th, itype):
		'''the type of i-th indiex is about to become itype'''
		if self.is_tensor:
			# prepare metric tensor and dummy index		
			if itype == 1:
				metric_type = 1
				dummy = 2*abs(self.indices).max()
			else:
				metric_type = 0
				dummy = -2*abs(self.indices).max()
			g = MetricTensor(metric_type)
			new_tensor = self.copy()		
			# generate new indices
			g.indices = np.array([-new_tensor.indices[i_th], -dummy])
			new_tensor.indices[i_th] = dummy
			new_tensor = g*new_tensor	
			self.indices[i_th] = -self.indices[i_th]	
			self[self.indices] = new_tensor
		else:
			raise IndexError("Tensor indices can't be lower or upper")
		
	def __getitem__(self, indices):
		indices = np.array(indices).flatten()
		if len(indices) != self.rank:
			raise IndexError("Indices don't match the tensor rank")
		elif len(indices) != len(set(indices)):
			raise IndexError("Indices are repeated")
		# change dummy index pairs to unrepeated pairs
		change_list = []
		for i in xrange(self.rank):
			for j in xrange(i + 1, self.rank):
				if indices[i] == -indices[j]:
					indices[j] *= 100
					change_list.append(j) 
		new_tensor = self.copy()
		# generate default indices
		indices_default = abs(indices)
		for i in xrange(len(indices)):
			if new_tensor.ind_type[i] > 0:
				indices_default[i] = -indices_default[i]
		# upper and lower indices
		new_tensor.indices = indices_default
		for i in xrange(len(indices)):
			if new_tensor.ind_type[i] != (indices[i] < 0):
				new_tensor.updown(i, indices[i] < 0)
		# recover all changed indices
		for i in change_list:
			new_tensor.indices[i] /= 100
		return new_tensor.contraction()
			
	def __setitem__(self, indices, other):
		indices = np.array(indices).flatten()
		self.indices = indices
		if (self.is_tensor == False) and (False in (self.ind_type == (indices < 0).astype(int))):
			raise IndexError("Tensor indices can't be lower or upper")
		else:
			self.ind_type = (indices < 0).astype(int)
		if type(other) == Tensor:
			if (self.rank == other.rank) and (set(indices) == set(other.indices)):
				inds_list_self = TensorIndex(self.rank)
				change_list = [0]*self.rank
				for i in xrange(self.rank):
					for j in xrange(self.rank):
						if indices[i] == other.indices[j]:
							change_list[j] = i
				inds_list_other = inds_list_self[:, change_list]
				for i in xrange(len(inds_list_self)):
					ind_self = tuple(inds_list_self[i])
					ind_other = tuple(inds_list_other[i])
					self.value[ind_self] = other.value[ind_other]
			else:
				raise IndexError("Inices don't match")
		else:
			raise ValueError("Target can only be tensor")
				
	def der(self, *indices):
		dim = len(metric)
		indices = np.array(indices).flatten()
		if (indices < 0).sum() == 0:
			raise IndexError("Only derivatives of 1-form are allowed")
		elif set(self.indices).intersection(set(indices)) != set():
			raise IndexError("Indices are repeated")
		new_rank = self.rank + len(indices)
		new_indices = np.append(self.indices, indices)
		new_type = (new_indices < 0).astype(int)
		inds_list = TensorIndex(new_rank)
		new_value = np.empty(new_rank*[dim], object)
		for i in xrange(len(inds_list)):
			ind_new = tuple(inds_list[i])
			ind_self = tuple(inds_list[i][:self.rank])
			ind_der = list(inds_list[i][self.rank:])
			new_value[ind_new] = sym.Derivative(self.value[ind_self], *basis[ind_der], 
                                                evaluate = True)
		new_tensor =  Tensor(new_type)
		new_tensor.indices = new_indices
		new_tensor.value = new_value
		return new_tensor.contraction()
		
	def __repr__(self):
		return self.value.__str__()
		
		

