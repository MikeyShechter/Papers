# Table of Contents

- [Table of Contents](#table-of-contents)
- [Papers](#papers)
  - [Multi-Task Learning](#multi-task-learning)
    - [An Overview of Multi-Task Learning in Deep Neural Networks (05.2017)](#an-overview-of-multi-task-learning-in-deep-neural-networks-052017)

# Papers

## Multi-Task Learning

### An Overview of Multi-Task Learning in Deep Neural Networks (05.2017)

- Sebastian Ruder
- [blog-post](https://ruder.io/multi-task/)
- [paper](https://arxiv.org/pdf/1706.05098.pdf)
This blog-post is an overview of multi-task learning papers.  
I will link to the original papers, and when/if I read them, I will summarize them here as well.

Multi-task learning is when you optimize multiple tasks in one model.  

MTL methods:

- Hard parameter sharing:  
  Sharing some hidden layers between all tasks - like in MOMO.
- soft parameter sharing:  
  Each task has different parameters, but the distance between them is regularized to make the parameters similar.  
  For example $l_2$ norm or [Trace Norm](https://arxiv.org/abs/1606.04038)  

Reasons why MTL works:

- Learning multiple tasks enables learning a better representation of the data.
- Attention focusing - MTL helps a model focus its attention on features that matter, as other tasks provide evidence for the relevance of those features.
- Eavesdropping - Some features F that are important for a task A might be difficult to learn directly for task A. Maybe because F interacts with A in a complex way.  
  With MTL you can learn a task B which makes learning F easier, or even directly learn F, a method called [Learning from hints](https://www.sciencedirect.com/science/article/pii/0885064X9090006Y?via%3Dihub)
- Representation Bias - MTL biases the model to prefer representations that other tasks also prefer. This will help the model generalize for other tasks as well as long as they are from the same [environment](https://arxiv.org/abs/1106.0245)