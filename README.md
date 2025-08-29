# Investigating the Outside Option in RLHF reward models
Code and notes exploring RLHF reward models. Details a failed attempt to use an 'outside option' in preference labeling, leading to a deeper understanding of calibration and experimental design. 🧪

---
## TL;DR
This project explores using an "outside option" in RLHF to learn an absolute reward scale. The investigation found this approach is **theoretically incompatible** with modern frameworks like DPO and **practically redundant** with older methods like PPO. A naive experiment that simply removes "bad" data is also identified as a flawed evaluation method that produces misleadingly high scores.

The project concludes that the most valuable use for an "outside option" is not as a complex modeling feature, but as a practical **data curation heuristic** to improve the quality of the training dataset.

---
## The Hypothesis

### 1. The Hidden Assumption in Standard RLHF

Standard Reinforcement Learning from Human Feedback (RLHF) and its recent advancements are built on a critical implicit assumption: **that the chosen response in a pairwise comparison is a desirable one.** This framework is designed to determine which of two options is *better*, effectively treating the selected response as a positive example. For instance, papers discussing Binary Classifier Optimisation (BCO) and Kahneman-Tversky Optimization (KTO) explicitly state that the chosen response is a desirable one.

### 2. Why This Assumption Fails in Practice

This assumption frequently breaks down. Despite dramatic improvements, large language models still suffer from significant issues like **hallucination**, factual inaccuracies, and unhelpful outputs. Consequently, it is common for a human labeler to be presented with a choice where **both responses are undesirable**.

In a forced-choice system, the labeler can only select the "less bad" option. This trains the model to be marginally better than a poor alternative, rather than teaching it what constitutes an objectively good response.

### 3. The Proposed Solution: An "Outside Option"

To address this gap, this project hypothesizes that introducing an **"outside option"**—allowing the labeler to reject both responses—can fundamentally improve the training signal.

This choice provides unambiguous negative feedback, explicitly teaching the reward model which types of outputs to avoid. The goal is to move beyond learning simple relative preferences and instead train a model on an **anchored, absolute scale of quality**, enabling it to distinguish more effectively between good and bad responses.

Of course. Here is a draft for the theoretical methodology section. It begins with the standard RLHF and DPO frameworks and then introduces your choice model as a more expressive alternative.

---
## Methodology: From Relative Preferences to an Absolute Scale

### 1. The Standard Framework: RLHF and Direct Preference Optimization (DPO)

The standard approach to aligning language models with human preferences involves a three-stage process: Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Reinforcement Learning (RL).

The **Reward Modeling** stage typically uses the **Bradley-Terry model**, which assumes human preferences follow a logistic function of the difference between the rewards of two responses, $y_w$ (winner) and $y_l$ (loser):

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

The recent **Direct Preference Optimization (DPO)** paper streamlined this process by showing that one can directly optimize the language model policy on preference data, bypassing the explicit reward modeling stage. DPO's objective function is also derived from this same Bradley-Terry model, meaning it is fundamentally based on the **relative difference** between a chosen and a rejected response.

A key limitation of these models is that they are unanchored—they can't distinguish between a choice of "good vs. great" and "bad vs. terrible."

### 2. A Choice-Based Model with an Absolute Anchor

To overcome this limitation, we reformulate the problem using a **Random Utility Model**, a standard framework from consumer choice theory. Instead of a simple pairwise choice, we provide the human labeler with three options:
* $y_1$: Response 1 from the LLM.
* $y_2$: Response 2 from the LLM.
* $y_0$: An **"outside option"** to reject both responses.

We assume the labeler chooses the option $i$ that maximizes their latent utility, which consists of a deterministic reward component $r(x, y_i)$ and a random noise component $\epsilon_i$:

$$\text{Choose } i \text{ if } r(x, y_i) + \epsilon_i \geq \max_{j \in \{0,1,2\}} (r(x, y_j) + \epsilon_j)$$

### 3. Mathematical Formulation

By modeling the noise terms $\epsilon_i$ as following an i.i.d. Gumbel distribution, the probability of choosing option $i$ becomes the familiar softmax function over the rewards:

$$P(\text{choose } i | x, y_1, y_2) = \frac{\exp(r(x, y_i))}{\sum_{j=0}^{2} \exp(r(x, y_j))}$$

Crucially, we set the reward for the "outside option" to be a fixed constant, creating an **anchor** for the entire reward scale. For simplicity, we normalize this to zero:

$$r(x, y_0) = 0$$

This anchor allows us to identify the absolute reward of any given response directly from the choice probabilities. By taking the ratio of the probability of choosing response $y_i$ to the probability of choosing the outside option $y_0$, we get:

$$\frac{P(\text{choose } i)}{P(\text{choose } 0)} = \frac{\exp(r(x, y_i))}{\exp(r(x, y_0))}$$

Taking the logarithm and substituting $r(x, y_0) = 0$, we arrive at the result:

$$\log\left(\frac{P(\text{choose } i)}{P(\text{choose } 0)}\right) = r(x, y_i)$$

This means we can train a reward model $r_\theta(x,y)$ using a standard cross-entropy loss on the three-way choice data, and the resulting model will learn an **absolute, anchored reward** for any given response. This provides a much richer and more stable signal for alignment than a purely relative one.

### 4. The DPO Perspective: Redefining the Reward Function

For comparison, we can look at the approach taken by Direct Preference Optimization (DPO). DPO also reformulates the problem to avoid explicit reward modeling, but does so by redefining the reward function itself.

The derivation starts with the standard closed-form solution for the optimal policy $\pi$ that maximizes reward while being constrained by a KL-divergence penalty from a reference policy $\pi_{ref}$:

$$\pi(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

DPO's key insight is to algebraically invert this equation to solve for the reward function $r(x,y)$. This redefines the reward in terms of the policies:

$$r(x,y) = \beta \log\left(\frac{\pi(y|x)}{\pi_{ref}(y|x)}\right) + \beta \log(Z(x))$$

---
## Problematic Issues with Choice Model with Outsdie Option.

### Model 1: DPO

The model depends on the **difference** in rewards.

$$r(y_w) - r(y_l) = \left[\beta \log\frac{\pi(y_w)}{\pi_{ref}(y_w)} + C(x)\right] - \left[\beta \log\frac{\pi(y_l)}{\pi_{ref}(y_l)} + C(x)\right]$$

The prompt-dependent term $C(x) = \beta \log(Z(x))$ **cancels out**.
---
### Model 2: Suggested Choice Model with outside option

The model uses a softmax over three options, where the reward for the outside option is a fixed anchor, $r(y_0) = 0$. The DPO reward definition is applied only to the LLM responses $r(y_1)$ and $r(y_2)$.

Because the constant term $C(x)$ is added to the rewards for $y_1$ and $y_2$ but **not** to the reward for the anchor $y_0$, it **DOES NOT CANCEL OUT** from the softmax calculation.

**Implication:** The DPO reward redefinition is incompatible with a fixed-anchor choice model because the intractable partition function $Z(x)$ remains in the loss function.

Of course. Here is a draft for the "Issues" section, restructured from your notes for clarity.

---
## Theoretical Issues and Limitations

While the choice-based model with an outside option provides a compelling theoretical framework for learning an absolute reward scale, it faces two significant issues when considered in the context of modern alignment techniques like DPO or traditional PPO.

### 1. The C(x) Estimation and Identifiability Problem

When the DPO reward definition is substituted into the choice model, the prompt-dependent term $C(x) = \beta \log(Z(x))$ does not cancel out. This presents a major obstacle:

* **Intractability:** The partition function $Z(x)$ sums over the entire vocabulary space for all possible response lengths, making it intractable to compute directly.
* **Identifiability:** Any attempt to parameterize and learn $C(x)$ would create a severe identifiability problem. The term $C(x)$ is a function of the prompt $x$ only. In an autoregressive model, the log-probability of the first generated token, $\log \pi(y_1|x)$, is also a function of $x$ only. A neural network would struggle to distinguish between these two components, as they are perfectly collinear. This training instability would make it difficult to learn a meaningful reward function.

### 2. Redundancy of the Outside Option in an RM + PPO Pipeline

If we set aside the DPO formulation and consider the traditional two-stage pipeline (Reward Modeling followed by RL with PPO), the theoretical advantage of the outside option diminishes.

* **What is Learned:** A standard pairwise comparison model identifies a reward function up to a constant (e.g., $r(y) + C$). The choice model with an outside option can identify the reward function's absolute scale by fixing this constant.
* **What PPO Uses:** The PPO optimization objective is to find a policy that maximizes expected reward. This optimization is **invariant** to adding a constant to the reward function. The policy that is optimal for a reward $r(y)$ is the same policy that is optimal for $r(y) + C$.

Therefore, the extra information about the absolute scale of the reward—the primary benefit of the choice model—is theoretically discarded during the RL fine-tuning stage. This suggests that the added complexity of the three-way choice model may be redundant in a standard RM+PPO setup.

---
## Experimental Setup

### 1. Base Model and Dataset
The experiment used the **`google/gemma-3-270m`** model as the base. The training and evaluation data were sourced from the **`stanfordnlp/imdb`** dataset available on Hugging Face, which contains movie reviews labeled for sentiment.

### 2. Supervised Fine-Tuning (SFT)
An SFT baseline model was created by fine-tuning the Gemma model on the last 12,000 positive samples from the IMDB training set. For each sample, the first 20 tokens (excluding the BOS token) were treated as the **prompt**, and the remaining text was treated as the desired **response**. This step trained the base model to generate positive-sentiment text.

### 3. Preference Data Simulation
To create a preference dataset for DPO training, the first 12,000 negative samples of the IMDB training set were used. For each sample, the first 20 tokens were used as prompts.
1.  For each prompt, a pair of responses was generated using the SFT model with default generation settings.
2.  The **`cardiffnlp/twitter-roberta-base-sentiment-latest`** model was used as a ground-truth reward model to score the positivity of each generated response.
3.  A three-way preference was then simulated. An "outside option" with a fixed reward of 0 was introduced. Gumbel noise was added to the rewards of both responses and the outside option, and the final preference was determined by the highest resulting score. This simulates a human labeler choosing the best option among response 1, response 2, or neither.

---
## Models Compared
Four models were trained and evaluated:

1.  **SFT (Baseline):** The supervised fine-tuned model with no subsequent preference alignment.
2.  **SFT + DPO with Full Data:** The SFT model further trained with DPO on the preference data. It was trained on the pairwise preferences between the two generated responses, ignoring any preference involving the outside option.
3.  **SFT + DPO with Data Removal:** The SFT model trained with DPO on a **filtered dataset**. All samples where the "outside option" was the most preferred choice were removed. This simulates the flawed methodology discussed previously.
4.  **SFT + DPO with Outside Option:** The SFT model trained with a custom DPO-like objective that attempts to incorporate the three-way preference, including the outside option. This model attempts to learn the absolute reward scale and implicitly estimates the partition function $Z(x)$.

---
## Evaluation & Results

Each of the four models was used to generate responses from prompts in the IMDB evaluation set. The generated texts were then scored using the same ground-truth RoBERTa reward model.

| Model                                    | Mean Reward | Standard Deviation |
| :--------------------------------------- | :---------: | :----------------: |
| **SFT + DPO with Data Removal** |  **0.2392** |      0.6789      |
| SFT + DPO with Full Data (Standard DPO)  |   0.1527    |      0.6882      |
| SFT + DPO with Outside Option (Custom)   |   0.0632    |      0.7008      |
| SFT (Baseline)                           |  -0.2047    |      0.7144      |

---
## Interpretation

The results provide clear empirical evidence for the theoretical issues discussed.

1.  **Standard DPO is Effective:** The **SFT + DPO with Full Data** model (Mean Reward: 0.1527) shows a significant improvement over the **SFT baseline** (Mean Reward: -0.2047). This confirms that standard DPO is effective at aligning the model's outputs with the preferences of the reward model.

2.  **Data Removal is Misleadingly Successful:** The **SFT + DPO with Data Removal** model achieved the highest score (Mean Reward: 0.2392). This result perfectly demonstrates the **survivorship bias** flaw. By removing the most challenging samples where both generated responses were poor, the model was trained on an easier, pre-filtered dataset. It learned to perform exceptionally well on "easy" cases but was never trained on the difficult ones, leading to an inflated and misleading evaluation score.

3.  **The Custom Model Fails as Predicted:** The **SFT + DPO with Outside Option** model performed the worst of all the aligned models, showing only a marginal improvement over the SFT baseline. This result strongly supports the theoretical issues raised earlier. The model, which attempts to learn the intractable partition function $Z(x)$, likely suffers from an **unstable training process and the identifiability problem**, failing to converge to a good policy.

In conclusion, the experiment successfully demonstrates that while standard DPO is a robust alignment method, the flawed approach of removing "bad" data leads to deceptively high scores, and attempting to learn an absolute reward scale via a custom DPO objective is practically ineffective.
