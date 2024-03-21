def step(self, hidden_states, conv_state, ssm_state):
    # Ensure input hidden states are of expected dimensionality
    dtype = hidden_states.dtype
    assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
    
    # Project hidden states and split into two parts: 'x' and 'z'
    xz = self.in_proj(hidden_states.squeeze(1))  # Reduce dimensionality and project
    x, z = xz.chunk(2, dim=-1)  # Split into two parts for different uses in the network
    
    # Convolutional (Conv) step
    if causal_conv1d_update is None:
        # Update convolutional state, shifting and inserting new 'x'
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        # Apply convolution using current state, followed by activation function
        x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None:
            x += self.conv1d.bias
        x = self.act(x).to(dtype=dtype)  # Apply activation function
    else:
        # Alternative convolutional update if provided
        x = causal_conv1d_update(
            x, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, self.activation
        )
    
    # Further project 'x' for state space model (SSM) processing
    x_db = self.x_proj(x)  # Project 'x' for SSM usage
    # Split the result into time, state, and control components
    dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    
    # State Space Model (SSM) step
    if selective_state_update is None:
        # Process time increment and update SSM matrices
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))  # Process time increments
        A = -torch.exp(self.A_log.float())  # Create system matrix 'A'
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # Discretize 'A'
        dB = torch.einsum("bd,bn->bdn", dt, B)  # Discretize 'B'
        # Update SSM state based on model dynamics
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        # Calculate output 'y' based on SSM state and input
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y += self.D.to(dtype) * x  # Add direct path from 'x' to output
        y *= self.act(z)  # Apply activation function to output
    else:
        # Alternative SSM update if provided
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        )
    
    # Final output projection
    out = self.out_proj(y)
    return out.unsqueeze(1), conv_state, ssm_state
