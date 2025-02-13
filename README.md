# humansam
abs:Numerous synthesized videos from diffusion models, especially human-centric ones that simulate realistic human actions, pose significant threats to human information security and authenticity. While progress has been made in binary forgery video detection, it remains under-explored to comprehensively study the specific forgery types. In this paper, we propose HumanSAM, to further classify human-centric forgeries into three specific types of artifacts commonly observed in generated content: spatial, appearance and motion anomaly. To better capture the features of geometry, semantics and spatiotemporal consistency, we propose to generate the human forgery representation by fusing two branches of video understanding and spatial depth. We also adopt a rank-based confidence enhancement strategy during the training process to learn more robust representation by introducing three prior scores.   
For training and evaluation, we construct the first public benchmark, the Human-centric Forgery Video (HFV) dataset, with all types of forgeries carefully annotated. 
In our experiments, HumanSAM yields promising results in comparison
with state-of-the-art methods, both in binary and multi-class forgery classification.

eval_dataset:https://huggingface.co/datasets/humansam/cogvideox5b_human_dim
