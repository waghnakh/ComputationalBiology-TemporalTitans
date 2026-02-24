# Explanation of this assignment

Hey guys, if you're looking at the code and wondering here is the breakdown of the biology and the math. I used a bit of AI for this explanation so kindly excuse me haha!

## Protein A and Protein B
Think of the cell like a car.
* **Protein A (Tumor Suppressor):** These are the **Brakes**. Its entire job is to stop the cell from dividing out of control.
* **Protein B (Oncogene):** This is the **Gas Pedal**. Its job is to make the cell grow and divide. 

In a healthy cell, Protein A keeps Protein B in check. In our two cancer patients (Alpha and Beta), the brakes are broken, but they are broken in two entirely different ways.

---

## What are Mechanism 1 and Mechanism 2?
To make Protein B (the Gas Pedal), the cell has to print an instruction manual (mRNA). It prints a "Rough Draft" (pre mRNA with junk code called *introns*) and then uses molecular scissors to cut out the junk and make a "Final Manual" (*exons*).

* **Mechanism 1 (Transcriptional Hijack):** The printer's off switch is broken. Protein A is supposed to turn the printer off, but it can't. The cell just prints perfect, finished manuals at maximum speed. 
* **Mechanism 2 (Splicing Sabotage):** The printer works fine, but Protein A acts like an evil editor and hides the "scissors." Because the cell can't cut out the junk code, a massive mountain of useless Rough Drafts (pre-mRNA) piles up to the ceiling.

---

## Why we chose Mech 1 for Alpha and Mech 2 for Beta
How did we know which patient had which broken mechanism? We used the **Viterbi Algorithm** as a detective to look at RNA sequences.

* **Patient Alpha (`AGCGC`):** The algorithm calculated that this sequence is an **Exon** (a Final Manual). Because we found a Final Manual, we know the "scissors" are working perfectly. If the scissors work but the cell is still cancerous, the problem must be at the printer. **Therefore, Patient Alpha is Mechanism 1.**
* **Patient Beta (`AUUAU`):** The algorithm calculated that this sequence is an **Intron** (a Rough Draft). Because we found a Rough Draft floating around, we know the "scissors" are broken and the manuals aren't getting finished. **Therefore, Patient Beta is Mechanism 2.**

---

## Why we used ODE for Mech 1 and SDE for Mech 2
We couldn't use the same math for both patients because the physics of their factories are completely different.


* **Patient Alpha gets ODE (Ordinary Differential Equations):**
  ODEs are used for systems that flow smoothly and predictably. In Patient Alpha, the printer is stuck in the "ON" position, but the assembly line is flowing smoothly. There are no traffic jams. We can use smooth, predictable ODE math to show the steady, gradual takeover of the cancer.
* **Patient Beta gets SDE (Stochastic Differential Equations):**
  SDEs are used for systems that are noisy, chaotic, and have massive traffic jams. Because Patient Beta's "scissors" are broken, there is a giant pileup of unedited Rough Drafts. This creates an unstable ticking time bomb. We need the "Stochastic" (noisy/random) part of the math to simulate the exact, unpredictable microsecond the dam breaks, causing a sudden, explosive spike of Protein B!

---  

## Task C: The Diagnosis and Comparison
This task is just us running the code to plot the graphs from Task A and Task B side by side so we can compare the two patients.
* **Patient Alpha's Graph:** Shows a smooth, steady curve. This means the tumor is slow and stable.
* **Patient Beta's Graph:** Shows a terrifying vertical spike. This visually proves why Beta's cancer is way more aggressive and unpredictable it hoards RNA and then releases it all at once.

## Bonus Task: The Predator and Prey Loop


**What's happening:** After Protein B (the cancer) takes over, it needs to feed the tumor. It creates an Enzyme to act like a predator and eat the cell's Resources (the prey). It is literally the exact same math used to track wolves eating rabbits in a forest.
**The Math:** We coded the Lotka Volterra equations to see what happens to the cell in the long run. Our code calculates the exact "balance point" where the predator and prey are equal. 
**The Result:** By checking the eigenvalues, our code proves the cell gets trapped in an endless cycle. The resources grow, the enzymes eat them all and multiply, the resources run out, the enzymes starve, and the cycle repeats forever. Our stream plot maps out this endless loop perfectly.
