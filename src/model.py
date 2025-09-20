import numpy as np
import pickle
from typing import List, Tuple, Optional, cast


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function."""
    return (x > 0).astype(np.float32)


def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot initialization for weights."""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class Linear:
    """Linear (fully connected) layer."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize linear layer with Xavier initialization."""
        self.in_features = in_features
        self.out_features = out_features

        self.weight = xavier_init(in_features, out_features)
        self.bias = np.zeros(out_features, dtype=np.float32)

        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

        self.input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through linear layer."""
        self.input = x
        return x @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through linear layer."""
        batch_size = grad_output.shape[0]

        if self.input is not None:
            self.weight_grad = self.input.T @ grad_output / batch_size
        self.bias_grad = np.sum(grad_output, axis=0) / batch_size

        grad_input = grad_output @ self.weight.T
        return grad_input

    def zero_grad(self):
        """Zero out gradients."""
        self.weight_grad.fill(0)
        self.bias_grad.fill(0)


class MLP:
    """Multi-layer Perceptron with ReLU activations."""

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """Initialize MLP with given layer sizes.

        Args:
            layer_sizes: List of layer sizes, e.g., [input_dim, hidden1, hidden2, output_dim]
            learning_rate: Learning rate for SGD
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        self.layers: List[Linear] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.activations: List[Optional[np.ndarray]] = [None] * (len(self.layers) - 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        out = x

        for i, layer in enumerate(self.layers[:-1]):
            out = layer.forward(out)
            self.activations[i] = out.copy()
            out = relu(out)

        out = self.layers[-1].forward(out)

        return out

    def backward(self, loss_grad: np.ndarray) -> None:
        """Backward pass through the network.

        Args:
            loss_grad: Gradient of loss w.r.t. output, shape (batch_size, output_dim)
        """
        grad = loss_grad

        grad = self.layers[-1].backward(grad)

        for i in range(len(self.layers) - 2, -1, -1):
            if self.activations[i] is not None:
                grad = grad * relu_derivative(cast(np.ndarray, self.activations[i]))
            grad = self.layers[i].backward(grad)

    def step(self) -> None:
        """Update parameters using SGD."""
        for layer in self.layers:
            layer.weight -= self.learning_rate * layer.weight_grad
            layer.bias -= self.learning_rate * layer.bias_grad

    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for layer in self.layers:
            layer.zero_grad()

    def save(self, filepath: str) -> None:
        """Save model parameters to file.

        Args:
            filepath: Path to save the model
        """
        state = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'weights': [],
            'biases': []
        }

        for layer in self.layers:
            state['weights'].append(layer.weight.copy())
            state['biases'].append(layer.bias.copy())

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        """Load model parameters from file.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.layer_sizes = state['layer_sizes']
        self.learning_rate = state['learning_rate']

        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            layer = Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            layer.weight = state['weights'][i].copy()
            layer.bias = state['biases'][i].copy()
            self.layers.append(layer)

        self.activations = [None] * (len(self.layers) - 1)

    @staticmethod
    def from_file(filepath: str) -> 'MLP':
        """Create MLP instance from saved file.

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded MLP instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        model = MLP(state['layer_sizes'], state['learning_rate'])
        model.load(filepath)
        return model


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute cross-entropy loss and its gradient.

    Args:
        logits: Model outputs, shape (batch_size, num_classes)
        targets: Target class indices, shape (batch_size,)

    Returns:
        Tuple of (loss, gradient w.r.t. logits)
    """
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    log_probs = np.log(probs + 1e-10)

    targets_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    targets_one_hot[np.arange(batch_size), targets] = 1.0

    loss = -np.sum(targets_one_hot * log_probs) / batch_size

    grad = (probs - targets_one_hot)

    return loss, grad


class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """Initialize layer normalization."""
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = np.ones(normalized_shape, dtype=np.float32)
        self.bias = np.zeros(normalized_shape, dtype=np.float32)
        
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        
        self.input_normalized: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through layer norm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(var + self.eps)
        
        self.input_normalized = (x - mean) / self.std
        
        return self.input_normalized * self.weight + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through layer norm."""
        batch_size = grad_output.shape[0]
        
        self.weight_grad = np.sum(grad_output * self.input_normalized, axis=0) / batch_size
        self.bias_grad = np.sum(grad_output, axis=0) / batch_size
        
        N = self.normalized_shape
        
        grad_normalized = grad_output * self.weight
        
        grad_var = np.sum(grad_normalized * self.input_normalized, axis=-1, keepdims=True) * -0.5 / (self.std ** 3)
        grad_mean = np.sum(grad_normalized, axis=-1, keepdims=True) * -1 / self.std
        grad_mean += grad_var * np.mean(-2 * self.input_normalized * self.std, axis=-1, keepdims=True)
        
        grad_input = grad_normalized / self.std
        grad_input += grad_var * 2 * self.input_normalized * self.std / N
        grad_input += grad_mean / N
        
        return grad_input
    
    def zero_grad(self):
        """Zero out gradients."""
        self.weight_grad.fill(0)
        self.bias_grad.fill(0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, n_heads: int):
        """Initialize multi-head attention."""
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self.Q: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None
        self.attn_weights: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through multi-head attention."""
        batch_size, seq_len = x.shape[:2]

        Q_flat = self.W_q.forward(x)
        K_flat = self.W_k.forward(x)
        V_flat = self.W_v.forward(x)

        self.Q = Q_flat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        self.K = K_flat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        self.V = V_flat.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        self.scores = self.Q @ self.K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            self.scores = np.where(mask, self.scores, -1e9)

        self.attn_weights = softmax(self.scores, axis=-1)

        context = self.attn_weights @ self.V

        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        output = self.W_o.forward(context)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through multi-head attention."""
        batch_size, seq_len = grad_output.shape[:2]

        # Gradient through output projection
        grad_context = self.W_o.backward(grad_output)

        # Reshape for multi-head
        grad_context_reshaped = grad_context.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Gradient through attention: context = attn_weights @ V
        grad_attn_weights = grad_context_reshaped @ self.V.transpose(0, 1, 3, 2)
        grad_V = self.attn_weights.transpose(0, 1, 3, 2) @ grad_context_reshaped

        # Gradient through softmax
        grad_scores = grad_attn_weights * self.attn_weights
        grad_scores = grad_scores - self.attn_weights * np.sum(grad_scores, axis=-1, keepdims=True)

        # Scale gradient
        grad_scores = grad_scores / np.sqrt(self.d_k)

        # Gradient through scores = Q @ K^T
        grad_Q = grad_scores @ self.K
        grad_K = grad_scores.transpose(0, 1, 3, 2) @ self.Q

        # Reshape back
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Gradient through linear projections
        grad_input = self.W_q.backward(grad_Q) + self.W_k.backward(grad_K) + self.W_v.backward(grad_V)

        return grad_input
    
    def zero_grad(self):
        """Zero out gradients."""
        self.W_q.zero_grad()
        self.W_k.zero_grad()
        self.W_v.zero_grad()
        self.W_o.zero_grad()


class TransformerBlock:
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """Initialize transformer block."""
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        
        self.residual1: Optional[np.ndarray] = None
        self.residual2: Optional[np.ndarray] = None
        self.ff_hidden: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through transformer block."""
        self.residual1 = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x, mask)
        x = x + self.residual1
        
        self.residual2 = x
        x = self.norm2.forward(x)
        self.ff_hidden = relu(self.ff1.forward(x))
        x = self.ff2.forward(self.ff_hidden)
        x = x + self.residual2
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through transformer block."""
        grad = grad_output.copy()
        
        grad_ff2_input = self.ff2.backward(grad)
        grad_ff1_output = grad_ff2_input * relu_derivative(self.ff_hidden)
        grad_ff1_input = self.ff1.backward(grad_ff1_output)
        grad_norm2 = self.norm2.backward(grad_ff1_input)
        
        grad = grad + grad_norm2
        
        grad_attn = self.attention.backward(grad)
        grad_norm1 = self.norm1.backward(grad_attn)
        
        grad = grad + grad_norm1
        
        return grad
    
    def zero_grad(self):
        """Zero out gradients."""
        self.attention.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()
        self.ff1.zero_grad()
        self.ff2.zero_grad()


class Transformer:
    """Transformer model for sequence tasks."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int = 512):
        """Initialize transformer."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        self.embedding = xavier_init(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_len, d_model)
        
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, d_model * 4))
        
        self.output_proj = Linear(d_model, vocab_size)
        
        self.embedding_grad = np.zeros_like(self.embedding)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        pos_encoding = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding.astype(np.float32)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through transformer."""
        seq_len = x.shape[1]
        
        embedded = self.embedding[x]
        embedded = embedded + self.pos_encoding[:seq_len]
        
        for layer in self.layers:
            embedded = layer.forward(embedded, mask)
        
        output = self.output_proj.forward(embedded)
        
        return output
    
    def zero_grad(self):
        """Zero out gradients."""
        self.embedding_grad.fill(0)
        for layer in self.layers:
            layer.zero_grad()
        self.output_proj.zero_grad()
