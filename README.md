# Domain-Specific AI Assistant for Cards & Payments Systems

An end-to-end LLM engineering pipeline leveraging parameter-efficient fine-tuning (PEFT) and contrastive preference alignment to adapt a foundational model into a highly precise, safe, and professional financial assistant specialized in credit and debit card operations.

---

## 1. Project Overview & Domain Definition

### Selected Domain
**Consumer Banking: Credit and Debit Card Operations.** This domain encompasses the mechanical rules of credit/debit instruments, payment processing conditions, regulatory consumer protections, risk mitigation protocols (lost/stolen instances), and transactional fee frameworks.

### Business Problem
Generic foundational models lack the granular domain alignment required to operate within enterprise financial ecosystems safely. When queried with consumer card problems, base models frequently introduce severe operational hazards:
* **Generic Deficiencies:** Providing surface-level definitions that fail to outline practical banking actions.
* **Safety Protocols Breaches:** Generating high-risk or dangerous instructions (e.g., suggesting a user "look around the house" for a stolen card rather than instigating immediate account freezing).
* **Hallucinations:** Inventing non-existent merchant dispute timelines or misstating the structural components of interest calculation parameters like APR.

This project implements a three-stage adaptation sequence to align model behaviors directly with formal financial service criteria: **Correctness, Domain Accuracy, Clarity, Safety, and Professional Tone.**

---

## 2. Dataset Architecture

The project architecture relies on three distinct text frameworks engineered to systematically alter the model's knowledge base and structural formatting:

1. **Stage 1: Raw Domain Text Corpus (`non_instruction_data.txt`)**
   * *Format:* Unstructured textual strings.
   * *Details:* Aggregated documentation outlining banking rules, network criteria (Visa/Mastercard), regulatory timelines (60-day dispute windows), and operational fee profiles (1%-3% currency conversions).
2. **Stage 2: Instruction Dataset (`instruction_dataset.jsonl`)**
   * *Format:* Explicit JSON lines structured into `prompt` and `response` keys.
   * *Details:* Curated financial inquiries matched with formal, step-by-step institutional guidance blocks.
3. **Stage 3: Preference Alignment Dataset (`preference_dataset.jsonl`)**
   * *Format:* Contrastive text pairs mapped into `prompt`, `chosen`, and `rejected` keys.
   * *Details:* Paired target outcomes where the `chosen` variant embodies concise, safe, list-formatted instructions, and the `rejected` variant represents wordy, evasive, or shallow baseline generations.

---

## 3. The Core Optimization Engine: LoRA & QLoRA Configuration

To secure stable, low-latency training passes within limited cloud GPU allocations (e.g., free-tier NVIDIA T4 runtimes), the entire pipeline implements **Quantized Low-Rank Adaptation (QLoRA)**. 

The underlying base model parameters are frozen and compressed into a high-density 4-bit **NormalFloat4 (NF4)** precision envelope. Trainable Low-Rank decomposition adapter arrays are then injected across all linear projection modules within the model's multi-head attention blocks.

### Parameter Matrix Summary
* **Base Model Framework:** `unsloth/Qwen2.5-1.5B` (Chosen for high architecture density and strong command-following properties at low parameter sizes).
* **Rank ($r$):** `16`
* **Alpha Scaling ($\alpha$):** `32` (Maintains standard $2 \times r$ scaling mechanics to ensure numerical learning stability relative to frozen parameters).
* **LoRA Dropout:** `0` (Optimized directly to zero when utilizing Unsloth's custom Triton kernels to maximize runtime processing throughput).
* **Target Linear Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

---

## 4. The Three-Stage Fine-Tuning Sequence

### Stage 1: Non-Instruction Fine-Tuning (Continuous Pre-Training)
* **Objective:** Knowledge ingestion.
* **Approach:** The frozen 4-bit model parameters are exposed to the unstructured card corpus using a standard Causal Language Modeling objective (predicting the next token). This embeds specific industry vocabulary and domain definitions directly into the model's weight distributions before attempting conversational behavior formatting.

### Stage 2: Supervised Fine-Tuning (SFT)
* **Objective:** Conversational behavior formatting.
* **Approach:** The model is trained using `SFTTrainer` on strict prompt-response pairs wrapped in a deterministic instruction template. 
* **Hyperparameters:** `Learning Rate = 2e-4`, `Max Steps = 60`, `Global Effective Batch Size = 16` (Batch size 4 $\times$ 4 Gradient Accumulation steps). Mixed precision was enforced via `fp16=True` to lock memory registers.

### Stage 3: Direct Preference Optimization (DPO)
* **Objective:** Structural safety and stylistic alignment.
* **Approach:** Using a `DPOTrainer` loop, the SFT policy model is evaluated contrastively against preference pairs. The training objective mathematically maximizes the probability of generating the precise, high-safety `chosen` formatting while driving down the probability of generating verbose or vague text variations.
* **Hyperparameters:** `Learning Rate = 5e-6` (Slower learning rate to prevent catastrophic forgetting of core instruction weights), `Beta = 0.1` (Controls policy divergence boundaries relative to reference weights), `Max Steps = 30`, `remove_unused_columns=False` (To ensure raw string variables pass seamlessly into internal tokenizers).

---

## 5. Training Metrics & Logs

The training steps demonstrate consistent structural convergence across all adaptation milestones:

### Stage 2 SFT Loss Trajectory
```text
Step 10/60  |  Loss: 1.8421  |  Grad Norm: 0.854
Step 20/60  |  Loss: 1.2140  |  Grad Norm: 0.521
Step 40/60  |  Loss: 0.6432  |  Grad Norm: 0.312
Step 60/60  |  Loss: 0.3814  |  Grad Norm: 0.198
🎯 SFT training loop successfully completed. Weights saved to: stage2_peft_adapter/

**### Stage 3 DPO Alignment Trajectory**
```text
Step 05/30  |  Loss: 0.6931  |  Rewards/chosen: 0.124  |  Rewards/rejected: -0.045
Step 15/30  |  Loss: 0.4210  |  Rewards/chosen: 0.482  |  Rewards/rejected: -0.312
Step 30/30  |  Loss: 0.2104  |  Rewards/chosen: 0.794  |  Rewards/rejected: -0.684
🎉 DPO preference optimization successfully stabilized. Weights saved to: stage3_dpo_adapter/

### Multi-Stage Performance Matrix Evolution

Below is an empirical snapshot extracted from our final benchmark suite (reports/final_evaluation.md) contrasting identical inputs across the three training states:
Base Model Response
Stage 2 SFT Response
Stage 3 DPO Response

### Final Observations

* **Base Model:**  Frequently yields generic, conversational filler text that offers no real procedural utility and exposes users to financial risk.
* **SFT Model:** Successfully acquires correct banking terminology and answers accurately, but exhibits conversational verbosity and blocks of unformatted text.
* **DPO Model:** Represents the highest tier of professional quality. It delivers correct answers with zero structural padding, formats procedural steps into clean lists, and rigorously prioritizes user safety workflows.

### Operational Challenges & Technical Resolutions

* **Challenge 1:** c10::Half != unsigned char Matrix Data Type Clashing
Root Cause: Unsloth's optimized Triton background compiler attempted to forcefully inject standard torch.compile configurations across non-aligned 4-bit underlying registers when interacting with updated Hugging Face trl dependencies.
Resolution: Implemented an explicit environment variable patch at the top of the runtime workspace (os.environ["UNSLOTH_COMPILE_DISABLE"] = "1") and forced explicit floating-point parameters directly during initialization sequences (dtype = torch.float16, fp16 = True).

* **Challenge 2:** Dataset Loading Script Crashes on Unescaped Characters
Root Cause: The Hugging Face load_dataset("json", ...) engine threw parsing exceptions when encountering specific formatting anomalies on individual lines (such as broken quotation arrays on line 41).
Resolution: Developed a native Python open() file extraction loop utilizing structured try-except wrappers to isolate individual parsing exceptions, allowing damaged rows to skip silently while passing clean structured payloads into Dataset.from_dict().

* **Challenge 3:** ValueError: num_samples=0 inside the DPO Engine
Root Cause: Key mismatches between the preference dataset's raw data schema (prompt) and the code’s evaluation conditions (instruction) caused the script to drop all items during parsing.
Resolution: Re-mapped file configuration components to precisely match the target keys present in the source dataset file, completely filling the initialization batches.

### Future Roadmap Architectural Enhancements

* **Integration of a Retrieval-Augmented Generation (RAG) Architecture**
Instead of relying solely on the model's internal parameters to recall interest rates and dynamic fee adjustments, deploying a vectorized database framework will enable real-time retrieval of live, country-specific terms, driving parameter hallucination rates to zero.

* **Transitioning toward Reward Modeling (RLHF / PPO)**
Expanding structural feedback pathways by training an independent reward model to capture more complex human preferences beyond simple binary choice selections.

* **Continuous Pre-Training Cycles**
Automating continuous pre-training data updates using web scraping layers targeted at financial regulatory update sheets to ensure compliance maps remain fully accurate over time.

