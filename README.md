## Installation Instructions

To set up a conda environment and install this forked version of the `transformers` library, follow these steps:

1. **Create and activate a new conda environment**:

   ```bash
   conda create --name hpml python=3.10.16
   conda activate hpml
   ```

2. **Install the forked repository**:
 (todo: change the branch name)
   ```bash
   pip install git+https://github.com/jychen630/transformers@junyao_fast_single_template_constraint
   ```

3. **Install other dependencies**:

   ```bash
   pip install wandb torch
   pip install 'accelerate>=0.26.0'
   ```
