Most recent key files:
Jupyter Notebook: run-processor-1-2025
Processor: msdprocessor5

You can modify the betas/zcut range you want to generate histograms over in run-processor-1-2025. 

You can also modify the n - variable which determines how finely you want to split your beta/zcut range generation. For example, n=3 with beta=0.1 z_cut = 0.2 Generates:

Processor(n, beta, z_cut) = Hist(beta = 0.1/n, z_cut = 0.2/n) for all n from 0 to n.
