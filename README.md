# visual-llm

Project for MATS Winter 24-25 Neel/Arthur Stream.

For replication:

Create a conda environment using the yml file and separately git clone [SAEDashboard](https://github.com/jbloomAus/SAEDashboard) and [LLaVA](https://github.com/haotian-liu/LLaVA) (additional setup required for LLaVA, see their repo for details).  A modification has to be made to LLaVA source in order to perform feature visualization.  Namely, the torch.no_grad() decorator for the CLIP encoding forward pass must be removed to permit the gradient to pass (see relevant line [here](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/multimodal_encoder/clip_encoder.py#L45)). 

Code performs the following:

1) Makes Llava Mistral and Mistral-IT-v0.2 compatible with TransformerLens.
2) Trains sparse autoencoders (SAEs) for both models using SAELens.
3) Creates feature dashboards of those SAEs using SAEDashboard.
4) Generates automated explanations and interpretability scores using automated-interpretability (lifted code from neuronpedia_intergration.py in SAELens).
5) Generates feature visualization images for Llava neurons and SAE features using Lucent.
6) Analyzes data and generates plots.