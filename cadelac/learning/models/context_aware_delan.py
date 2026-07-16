import torch
import torch.nn as nn
import torch.autograd.functional as F

import numpy as np
import copy

class ComponentNN(nn.Module):
    def __init__(self, n_input, n_ouput, **kwargs):
        super(ComponentNN, self).__init__()

        # Read optional arguments:
        self.net_arch = kwargs.get("net_arch", None)
        self.n_dof = kwargs.get("n_dof", 2)
        self.n_enc_input = kwargs.get("n_enc_input", 1)
        self.activation_name = kwargs.get("activation", 'Tanh')
        self.apply_tf = kwargs.get("init_tf", True)
 
        self.n_output = n_ouput

        if self.activation_name == 'Tanh':
            self.act = nn.Tanh()
        elif self.activation_name == 'ReLu':
            self.act = nn.ReLU()
        else:
            raise AssertionError

        ## Create Networks
        self.layers = []
        # Create Input Layer
        input_layer_size = n_input
        if self.apply_tf:
            input_layer_size = 2 * input_layer_size

        if self.n_enc_input > 1:
            input_layer_size += self.n_enc_input

        prev_size = input_layer_size
        for hidden_size in self.net_arch:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))  # Linear layer
            self.layers.append(self.act)  # Activation function
            prev_size = hidden_size  # Update previous layer size
        # Create Output Layer
        self.layers.append(torch.nn.Linear(prev_size, self.n_output))
        
        self.net = nn.Sequential(*self.layers)
        self.net.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def input_tf(self, input):
        return torch.cat([torch.cos(input), torch.sin(input)], axis=-1)

    def forward(self, input, one_hot):
        
        if self.apply_tf:
            input = self.input_tf(input)

        if self.n_enc_input > 1:
            input = torch.cat((input, one_hot), dim=-1)

        return self.net(input)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        
        # Define a stacked LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to produce the output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward propagate through LSTM
        out, (hn, cn) = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Pass through the fully connected layer (output at the last time step)
        out = self.fc(out[:, -1, :])  # Use the last hidden state for output prediction
        return out
    
class ContextAwareDeLaN(nn.Module):
    def __init__(self, n_dof, **kwargs):
        super(ContextAwareDeLaN, self).__init__()

        self.n_dof = n_dof
        self.n_enc_input = kwargs.get("n_enc_input", 1)
        self.n_lstm_hidden = kwargs.get("n_lstm_hidden", 1)
        self.n_lstm_input = kwargs.get("n_lstm_input", 1)
        self.n_lstm_depth = kwargs.get("n_lstm_depth", 1)
        self.hist_length = kwargs.get("hist_length", 1)
        self.activation_name = kwargs.get("activation", 'Tanh')
        self.epsilon = kwargs.get("diagonal_epsilon", 1.e-5)
        self.shift = kwargs.get("diagonal_shift", 0.0)
        self.softplus_beta = kwargs.get("softplus_beta", 1.0)
        self.apply_tf = kwargs.get("init_tf", True)
        self.act_ld_name = kwargs.get("act_ld", 'Softplus')

        # Use general values if not defined
        kwargs_inertia = copy.deepcopy(kwargs)
        kwargs_pot = copy.deepcopy(kwargs)
#changed 1
        kwargs_mlp = copy.deepcopy(kwargs)

        self.net_arch_inertia = None
        self.net_arch_inertia = kwargs.get("net_arch_inertia", self.net_arch_inertia)
        kwargs_inertia["net_arch"] = self.net_arch_inertia

        self.net_arch_pot = None
        self.net_arch_pot = kwargs.get("net_arch_pot", self.net_arch_pot)
        kwargs_pot["net_arch"] = self.net_arch_pot

#changed 2
        # Configuration for the direct torque MLP
        self.net_arch_mlp = kwargs.get("net_arch_mlp", [30, 20]) #model architecture (searches for hyperparameters or defaults to defined)
        kwargs_mlp["net_arch"] = self.net_arch_mlp #fix naming convention for class ComponentNN
        kwargs_mlp["init_tf"] = False #we do not need sin cos conversion as we use q_dot, q_ddot

        # Compute non-zero elements of L:
        self.l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        self.l_diag_size = self.n_dof
        self.l_lower_size = self.l_output_size - self.l_diag_size

        self.inertia_net = ComponentNN(self.n_dof, self.l_output_size, **kwargs_inertia)
        self.potential_net = ComponentNN(self.n_dof, 1, **kwargs_pot)
#changed 3
        self.torque_net = ComponentNN(3 * self.n_dof, self.n_dof, **kwargs_mlp) #the actual model PARAMS: input dim, output dim

        if self.hist_length > 0:
            self.lstm = LSTMModel(self.n_lstm_input, self.n_lstm_hidden, self.n_enc_input, self.n_lstm_depth)

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(self.l_output_size)], np.arange(self.l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)

        # Compute Matrix Indices
        self.tril_indices = np.tril_indices(self.n_dof)

        if self.act_ld_name == 'Softplus':
            self.act_ld = torch.nn.Softplus(self.softplus_beta)
        elif self.act_ld_name == 'ReLu':
            self.act_ld = torch.nn.ReLU()
#changed 5 to use our mlp model
       # self.dyn_model = self.dyn_model_hessian
        self.dyn_model = self.dyn_model_mlp
        # Define vmap methods
        if self.n_enc_input == 1:
            self.vmap_jacfwd_lower_tri_inertia = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True), in_dims=(0, None))
            self.vmap_jacfwd_potential_energy = torch.func.vmap(torch.func.jacfwd(self.potential_energy, argnums=0, has_aux=True), in_dims=(0, None))
            self.vmap_jacfwd_inertia_out = torch.func.vmap(torch.func.jacfwd(self.inertia_out, argnums=0, has_aux=True), in_dims=(0, None))

            self.vmap_jacfwd_lagrangian_full = torch.func.vmap(torch.func.jacfwd(self.lagrangian_fn, argnums=(0,1)), in_dims=(0, 0, None))
            self.vmap_hessian_lagrangian_full = torch.func.vmap(torch.func.hessian(self.lagrangian_fn, argnums=(0,1)), in_dims=(0, 0, None))

            self.vmap_jacfwd_lagrangian_q = torch.func.vmap(torch.func.jacfwd(self.lagrangian_fn, argnums=0), in_dims=(0, 0, None))
            self.vmap_hessian_lagrangian_q_qd = torch.func.vmap(torch.func.hessian(self.lagrangian_fn, argnums=(0, 1)), in_dims=(0, 0, None))
            self.vmap_hessian_lagrangian_qd_qd = torch.func.vmap(torch.func.hessian(self.lagrangian_fn, argnums=1), in_dims=(0, 0, None))

            self.vmap_hessian_lagrangian_qd = torch.func.vmap(torch.func.jacrev(torch.func.jacfwd(self.lagrangian_fn, argnums=1), argnums=(0,1)), in_dims=(0, 0, None))

        else:
            self.vmap_jacfwd_lower_tri_inertia = torch.func.vmap(torch.func.jacfwd(self.lower_tri_inertia_fn, argnums=0, has_aux=True))
            self.vmap_jacfwd_potential_energy = torch.func.vmap(torch.func.jacfwd(self.potential_energy, argnums=0, has_aux=True))
            self.vmap_jacfwd_inertia_out =  torch.func.vmap(torch.func.jacfwd(self.inertia_out, argnums=0, has_aux=True))

            self.vmap_jacfwd_lagrangian_full = torch.func.vmap(torch.func.jacfwd(self.lagrangian_fn, argnums=(0,1)))
            self.vmap_hessian_lagrangian_full = torch.func.vmap(torch.func.hessian(self.lagrangian_fn, argnums=(0,1)))

            self.vmap_jacfwd_lagrangian_q = torch.func.vmap(torch.func.jacfwd(self.lagrangian_fn, argnums=0))

            self.vmap_hessian_lagrangian_q_qd = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(self.lagrangian_fn, argnums=0), argnums=1))
            self.vmap_hessian_lagrangian_qd_qd = torch.func.vmap(torch.func.jacfwd(torch.func.jacrev(self.lagrangian_fn, argnums=1), argnums=1))
            self.vmap_hessian_lagrangian_qd = torch.func.vmap(torch.func.jacrev(torch.func.jacfwd(self.lagrangian_fn, argnums=1), argnums=(0,1)))

    def mass_matrix_fn(self, q, enc_input):
        output = self.inertia_net(q, enc_input)
        l_diagonal, l_off_diagonal = torch.split(output, [self.l_diag_size, self.l_lower_size], dim=-1)

        l_diagonal = self.act_ld(l_diagonal + self.shift) + self.epsilon

        # Assemble l and der_l
        L_vec = torch.cat((l_diagonal, l_off_diagonal), -1)[..., self._idx]


        L = torch.zeros((self.n_dof, self.n_dof)).to(self.device)
        tril_indices = torch.tril_indices(self.n_dof, self.n_dof)
        L = L.index_put((tril_indices[0], tril_indices[1]), L_vec)

        H = torch.matmul(L, torch.permute(L, (1,0)))

        return H
    
    def lower_tri_inertia_fn(self, q, enc_input):
        output = self.inertia_net(q, enc_input)
        l_diagonal, l_off_diagonal = torch.split(output, [self.l_diag_size, self.l_lower_size], dim=-1)

        # Ensure positive diagonal
        l_diagonal = self.act_ld(l_diagonal + self.shift) + self.epsilon

        # Assemble l
        l_vec = torch.cat((l_diagonal, l_off_diagonal), -1)[..., self._idx]

        l = torch.zeros((self.n_dof, self.n_dof)).to(self.device)
        tril_indices = torch.tril_indices(self.n_dof, self.n_dof)
        l = l.index_put((tril_indices[0], tril_indices[1]), l_vec)

        # Returning twice to use as aux varible over the jacobian
        return l, l

    def kinetic_energy(self, q, qd, enc_input):
        mass_mat = self.mass_matrix_fn(q, enc_input)
        return 1. / 2. * torch.matmul(qd.view(1, self.n_dof),  torch.matmul(mass_mat, qd.view(self.n_dof, 1)))

    def potential_energy(self, q, enc_input):
        V = self.potential_net(q, enc_input)
        return V, V

    def inertia_out(self, q, enc_input):
        l_raw = self.inertia_net(q, enc_input)
        return l_raw, l_raw

    def lagrangian_fn(self, q, qd, enc_input):
        e_kin = self.kinetic_energy(q, qd, enc_input)
        e_pot, _ = self.potential_energy(q, enc_input)
        return e_kin - e_pot

    def dyn_model_hessian(self, q, qd, qdd, enc_input = None):
        ####
        dLdq = self.vmap_jacfwd_lagrangian_q(q, qd, enc_input)
        d2L_dqddq, d2Ld2qd = self.vmap_hessian_lagrangian_qd(q, qd, enc_input)

        qdd_reshaped = qdd.view(-1, self.n_dof, 1)
        qd_reshaped = qd.view(-1, self.n_dof, 1)

        # Compute the predicted generalized force:
        tau_pred = torch.matmul(d2Ld2qd.squeeze(), qdd_reshaped).squeeze() + torch.matmul(d2L_dqddq.squeeze(), qd_reshaped).squeeze() - dLdq.squeeze()

        dEdt = torch.sum(qd * tau_pred, dim=1)

        return tau_pred, dEdt

#changed 4  MLP implementation
    def dyn_model_mlp(self, q, qd, qdd, enc_input=None):
        mlp_input = torch.cat((q, qd, qdd), dim=-1) #conects tensors along one dimension
        tau_pred = self.torque_net(mlp_input, enc_input) # the returned tau is predicted by the MLP (mlp input: 6 + z 10 = 16dim vector getting put into MLP) OUTPUT: 2 dim
        dEdt = torch.sum(qd * tau_pred, dim=1) #only used to get dEdt mechanical torque for fitting return value amount (2) of forward. As loss_power = false this is unused

        return tau_pred, dEdt


    def forward(self, q, qd, qdd, lstm_input = None):
        if self.hist_length == 0:
            out = self.dyn_model(q, qd, qdd)
        else:
            enc_input = self.lstm(lstm_input)
            out = self.dyn_model(q, qd, qdd, enc_input)
        tau_pred = out[0]
        dEdt = out[1]
        return tau_pred, dEdt

    def inv_dyn(self, q, qd, qdd, lstm_input = None):
        if self.hist_length == 0:
            out = self.dyn_model(q, qd, qdd)
        else:
            enc_input = self.lstm(lstm_input)
            out = self.dyn_model(q, qd, qdd, enc_input)
        tau_pred = out[0]
        return tau_pred
    
    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(ContextAwareDeLaN, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(ContextAwareDeLaN, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self
