
## Probabilistic Graphical Models
# Markov Process
is a stochastic process that satisfies the Markov property. A process satisfies the Markov property if one can make predictions for the future of the process based solely on its present state just as well as one could knowing the process's full history, hence independently from such history; i.e., conditional on the present state of the system, its future and past states are independent.

A state diagram for a simple example is shown in the figure on the right, using a directed graph to picture the state transitions. The states represent whether a hypothetical stock market is exhibiting a bull market, bear market, or stagnant market trend during a given week. According to the figure, a bull week is followed by another bull week 90% of the time, a bear week 7.5% of the time, and a stagnant week the other 2.5% of the time. Labelling the state space {1 = bull, 2 = bear, 3 = stagnant} 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Finance_Markov_chain_example_state_space.svg/800px-Finance_Markov_chain_example_state_space.svg.png)

the transition matrix for this example is:
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6cea2dc36a546e141ce2d072636dbf8a0005f235)


![](https://wikimedia.org/api/rest_v1/media/math/render/svg/df26d8a65d9997bd816356f0ebc532c46ea9a46c)
