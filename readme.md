**Status:** Maintenance (expect bug fixes and minor updates)

Multi-Agent TD3 + Experience Sharing for cooperative soccer
========================================================================

This repository contains our version of the basic Multi-Agent extension of TD3 + Experience Sharing, used in "Learning to Play Soccer From Scratch: Sample-Efficient Emergent Coordination Through Curriculum-Learning and Competition". The code allows simple training of policies of any the defined stages (1v0, 1v1, and 2v2) in Deepmind's MuJoCo-based simulated soccer environment, for which we developed a simplified wrapper, that includes the transformed observations and rewards used in our work (Link to wrapper: https://github.com/Pavan-Samtani/dm_soccer2gym).

This work was derived from an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

If you wish to know more about the original Spinning Up Project by OpenAI
-------------------------------------------------------------------------

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Multi-Agent TD3 + ES
---------------------------

If you reference or use these versions of Multi-Agent TD3 for DeepMinds MuJoCo soccer environment in your research, please cite:

```
@inproceedings{samtani2021learning,
  title={Learning to Play Soccer from Scratch: Sample-Efficient Emergent Coordination
    through Curriculum-Learning and Competition},
  author={Samtani, Pavan and Leiva, Francisco and Ruiz-del-Solar, Javier},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021}
}
```

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
