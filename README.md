# Efficient Control for Dynamical Systems

### Douka Styliani
#### Master Thesis ~ Universite Paris-Saclay
CNRS ~ LISN ~ Team DATAFLOT


## Abstract
This thesis delves into the concept of efficient control for complex dynamical systems by leveraging reinforcement learning. Within this context, various dynamical systems in physics, especially in the domain of computational fluid dynamics, require extensive computational power in the form of time-consuming simulations, often spanning multiple hours. Despite these challenges, state-of-the-art deep reinforcement learning algorithms require the collection of millions of samples from the transition function to derive an optimal control policy. These statements highlight the need to streamline the control of complex dynamical systems through reduced data requirements. This study investigates exploration techniques for reduced sample complexity in the offline setting of model-based reinforcement learning and it focuses on further improving time efficiency and sample complexity. Furthermore, an existing offline algorithm is adapted to be used in the online setting, being able to collect transitions from the environment in real-time. The overall aim of this research is to refine control strategies to facilitate the integration of reinforcement learning in complex environments. The results show promise toward significantly more sample-efficient exploration in the offline setting.


## Acknowledgments
I would like to thank Rémy Hosseinkhan Boucher for his contribution to this work and his assistance in multiple aspects of the project including critical assessments, brainstorming, visualizations, coding, and debugging, and my supervisors Lionel Mathelin and Onofrio Semerano for their guidance and invaluable feedback.


## Contribution
In this work, we contributed to the efficient control of complex dynamical systems through the refinement of the Active Bayesian Reinforcement Learning (BARL) algorithm.

- Conducted a detailed analysis of the different parts of the algorithm and investigated the bottlenecks of the method and the technicalities of the acquisition function. 
- Further improved the sampling complexity of BARL on two control environments, Pendulum and Lorenz. 
- Advanced towards adapting the algorithm for online training without compromising the quality of the policy.
- Attempted to adapt an experience replay model to work with a small optimal dataset. 
- Proposed solutions to overcome the factors that affect the efficiency and effectiveness of BARL.

These contributions emphasize the potential of reinforcement learning in revolutionizing efficient control strategies for complex dynamical systems.

## Future Work
There are several aspects that we have not yet explored. We did not investigate the impact of reducing the number of posterior functions to approximate the acquisition function. We expect a significant time acceleration but a reduction in performance as well.
The optimized version can be further improved. One possibility is to replace the uniform sampling with a Bayesian or adaptive design. For this purpose, we could use the knowledge that similar points do not provide a high expected information gain and sample far away from the concentration of the current dataset. Another option would be to acquire several points on each iteration instead of one, also keeping in mind not to select close-in-proximity points. This could decrease the required iterations and thus reduce the time of posterior sampling and MPC time. We can also replace the Gaussian Process with a neural network and examine the trade-off between sample complexity and time efficiency.

In terms of online training, the algorithm is not ready for deployment. We are aware that the fitting of the GP kernel should be executed offline and thus a solution should be found for online adaptation. We have also not accounted for the unavoidable delays when observing a real dynamical system. A solution to this could be to include the time taken to choose an action as part of the Gaussian Process input. This way we could model the transition function with one more parameter for the delay of the actuation frequency and also provide an extra input in the policy during inference. This also has the potential to improve the GP's understanding of the system when no control is applied as it would have to interpolate the transition between actuation frequencies. Current control policies are constrained by the actuation frequency used during training. Thus, introducing it as an input that can be adapted in real time can have a huge impact on control methods. This is a setting that we intend to investigate in the future.

