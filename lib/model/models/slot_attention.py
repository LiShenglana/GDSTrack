import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # Layer normalization
        self.norm_inputs = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # Gaussian initialization parameters for slots
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))

        # Linear projections for query, key, and value
        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        # GRU cell for slot updates
        self.gru = nn.GRUCell(slot_size, slot_size)

        # MLP for slot updates
        self.mlp = nn.Sequential(
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size)
        )

    def forward(self, inputs):
        # inputs shape: [batch_size, num_inputs, input_size]
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size]
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size]

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size]
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            inputs.size(0), self.num_slots, self.slot_size, device=inputs.device)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention mechanism
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size]
            q *= self.slot_size ** -0.5  # Normalization
            attn_logits = torch.matmul(k, q.transpose(-1, -2))  # Shape: [batch_size, num_inputs, num_slots]
            attn = F.softmax(attn_logits, dim=-1)

            # Weighted mean
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)  # Shape: [batch_size, num_slots, slot_size]

            # Slot update
            slots = self.gru(updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots += self.mlp(self.norm_mlp(slots))

        return slots
