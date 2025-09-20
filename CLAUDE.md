# Connect 4 RL Agent - Systems-Focused Implementation

## Project Overview
A performance-oriented Connect 4 agent using self-play RL, designed to showcase systems engineering skills for ML infrastructure roles. Focus is on speed, profiling, and scalability rather than achieving state-of-the-art playing strength.

## Directory Structure
```
connect4-rl/
├── src/
│   ├── game.py         # Vectorized Connect 4 engine
│   ├── model.py        # Neural network (small CNN)
│   ├── train.py        # Self-play training loop
│   └── profile.py      # Performance profiling utilities
├── tests/
│   └── test_game.py    # Game logic validation
├── benchmarks/
│   └── results.json    # Performance metrics
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Environment
The conda environment is called `connect4-rl`. We should use poetry for dependency management.

## Core Design Principles
- **Vectorization First**: All operations should handle batch_size games simultaneously
- **Profile Everything**: Every major function should have profiling decorators
- **Minimal Dependencies**: PyTorch, Numba, NumPy, and profiling tools only
- **Systems Focus**: Optimize for throughput (games/second) over playing strength

## Code Style & Patterns
- Use type hints for all function signatures
- Implement `@profile` decorators for performance-critical functions
- Prefer NumPy operations over loops for board manipulation
- Use torch.jit.script where beneficial for hot paths
- Keep batch dimensions as first axis consistently

## Implementation Guidelines

### Game Engine (game.py)
- Use uint64 bitboards for position representation
- Implement parallel move validation using NumPy broadcasting
- Batch terminal state detection
- No single-game methods - everything operates on batches

### Model Architecture (model.py)
- Small CNN: 4-6 layers, ~100k parameters max
- Input: (batch, 2, 6, 7) - separate planes for each player
- Output: Policy head (7 actions) + Value head (scalar)
- Use torch.compile() for inference optimization

### Training Loop (train.py)
- Implement simple self-play without MCTS initially
- Use multiprocessing.Pool for parallel game generation
- Shared memory for experience buffer
- Log games/second metric prominently

### Profiling (profile.py)
- Memory profiling: Track GPU/CPU usage per component
- Speed profiling: Time each phase (generation, training, evaluation)
- Bottleneck analysis: Identify top 3 slowest operations
- Output clean JSON for visualization

## Performance Targets
- **Minimum**: 1,000 games/second on single GPU
- **Target**: 10,000 games/second on single GPU
- **Memory**: < 2GB GPU memory for batch_size=1024
- **CPU Utilization**: > 80% during self-play generation

## Common Pitfalls to Avoid
- Don't implement single-game logic then vectorize - start vectorized
- Don't use Python loops for board operations
- Don't store full game histories in memory - use circular buffer
- Don't forget to disable gradient computation during self-play

## Testing Requirements
- Validate game rules with property-based tests
- Benchmark against baseline (random play)
- Assert vectorized ops match single-game reference
- Profile tests should fail if performance regresses >10%

## When Working on This Project
1. **Always measure first**: Run profiler before optimizing
2. **Document performance changes**: Include before/after metrics in commits
3. **Keep PR scope small**: One optimization per PR
4. **Test batch consistency**: Ensure batch ops produce same results as sequential

## References
- Bitboard representation: https://github.com/denkspuren/BitboardC4
- Self-play optimization: AlphaZero paper Appendix
- PyTorch profiling: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html`