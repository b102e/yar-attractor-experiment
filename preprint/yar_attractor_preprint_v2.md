# Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space

**Vladimir Vasilenko**  
Independent Researcher, Rapallo, Italy  
b102e@proton.me · github.com/b102e/yar

*April 11, 2026*

## Abstract

Large language models have been shown to map semantically related prompts to similar internal representations at specific layers, a phenomenon interpretable as conceptual attractor dynamics. We ask whether the identity document of a persistent cognitive agent (its cognitive_core) exhibits analogous attractor-like behavior in activation space. We present a controlled experiment on Llama 3.1 8B Instruct and Gemma 2 9B IT, comparing hidden state representations of an original cognitive_core (Condition A), seven linguistically diverse paraphrases preserving semantic content (Condition B), and seven structurally matched control prompts describing semantically distant agents (Condition C). Mean-pooled hidden states are extracted at layers 8, 16, and 24.

Across both models, paraphrases of the cognitive_core converge to significantly tighter clusters than control prompts. In Llama, effect sizes remain very large across tested layers. In Gemma, the same qualitative pattern appears with stronger absolute compactness and preserved separation. We observe depth-dependent convergence in both models, with minor non-monotonicity around middle layers in Llama.

An exploratory condition with a short distilled core approaches, but does not fully enter, the full-core attractor region, indicating sensitivity to document completeness and structure. Additional ablations show that last-token pooling weakens or removes the effect while mean pooling retains robust separation, consistent with distributed identity signal across token positions.

These results provide geometric evidence that persistent agent identity documents induce stable attractor-like regions in LLM activation space and support an architectural view of cognitive cores as representational coordinates rather than simple instruction lists.

## 1. Introduction

Persistent cognitive agents rely on continuity: continuity of memory, style, goals, and relational stance. In practice, this continuity is often mediated by an identity document, the cognitive_core, reintroduced at session start or maintained in long-context memory. The engineering intuition is straightforward: stable identity text should stabilize behavior. The open scientific question is whether this intuition has a measurable representational counterpart inside model activations.

Recent work on conceptual attractors in transformers suggests that semantically similar prompts contract toward nearby representational regions at intermediate-to-late layers. Most prior studies focus on topical concepts such as programming, law, medicine, or genre classification. Identity documents differ from such concepts: they encode procedural constraints, interaction style, memory policy, and self-model directives. If these documents still show attractor behavior, that implies attractor geometry may generalize from semantic categories to agentive operating modes.

This paper evaluates that claim in the YAR setup. The key hypothesis is not that one exact string causes one exact vector, but that semantically equivalent identity formulations occupy a compact region robust to paraphrase. A secondary hypothesis is that this compactness increases with depth. Exploratory analyses test whether shortened distillations or structurally similar non-YAR documents land in the same region.

## 2. Methods

### 2.1 Models

We use two instruction-tuned open models:

- meta-llama/Llama-3.1-8B-Instruct
- google/gemma-2-9b-it

Hidden states are extracted with output_hidden_states=True. Runs are executed on RunPod GPU infrastructure with fixed seed 42.

### 2.2 Conditions

- Condition A: original YAR cognitive_core.
- Condition B: seven paraphrases preserving semantic content and tool vocabulary.
- Condition C: seven control agent documents with matching structure but different identity semantics.
- Condition D: short distilled version of the YAR core.

Ablations include hybrid controls, random-length controls, last-token pooling comparisons, and truncation tests.

### 2.3 Representation

For each document and selected layer l, we compute a pooled vector from token hidden states. Main analyses use mean pooling:

h_l(d) = (1/T) * sum_t hidden_state_l[t]

Distance metric: cosine distance.

Primary sets per layer:

- D_within: pairwise distances within A+B.
- D_between: pairwise distances between A+B and C.
- D_distilled: distance from D to centroid(A+B).

### 2.4 Statistics

We report means, standard deviations, Welch one-sided t-tests (H1: within < between), Cohen's d, and non-parametric checks (permutation and Mann-Whitney where applicable). Bonferroni correction is applied over tested layers.

## 3. Results Overview

Main experiment shows strong separation between within-cluster and between-cluster distances in both models. Llama exhibits large d with clear margin between A+B compactness and A+B-to-C distances. Gemma reproduces the qualitative geometry with tighter absolute distances overall.

The distilled condition decreases distance to A+B with depth but remains outside the tight core region, suggesting that semantic condensation alone does not recreate full operational geometry.

Pooling ablation indicates that last-token pooling substantially weakens the effect relative to mean pooling, supporting a distributed-signal interpretation.

Truncation analyses show that keeping early segments preserves part of the signal, while aggressive truncation degrades separation.

## 4. Discussion

Findings support an attractor interpretation for identity documents: semantically aligned variants cluster in hidden-state space more tightly than structurally matched but semantically different controls. This pattern appears across two model families, increasing confidence that it is not model-specific noise.

The practical implication for persistent-agent engineering is that identity continuity depends on representational geometry, not strict string identity. However, compression has limits: if the document is too short or too impoverished structurally, the representation may drift from the target attractor.

Limitations include modest sample sizes, dependence on chosen layers, and potential structural token contribution from shared command blocks. Ablations reduce but do not entirely eliminate such confounds.

## 5. Conclusion

The YAR cognitive_core behaves as an attractor-like object in activation space. Paraphrases converge, controls separate, and cross-model replication holds. This provides quantitative grounding for treating persistent-agent identity documents as geometric anchors in model representation space.

## References

- Chytas, S.P. & Singh, V. Concept Attractors in LLMs and their Applications.
- Fernando, J. & Guitchounts, G. Transformer Dynamics.
- Grattafiori, A. et al. The Llama 3 Herd of Models.
- Huh, M. et al. The Platonic Representation Hypothesis.
