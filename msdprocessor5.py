import time

import coffea.processor as processor
from coffea.analysis_tools import Weights
from coffea.nanoevents import NanoAODSchema
import awkward as ak
import numpy as np
import fastjet
import hist.dask as dah
import hist

NanoAODSchema.warn_missing_crossrefs = False


class msdProcessor(processor.ProcessorABC):
    def __init__(self, isMC=False):
        ################################
        # INITIALIZE COFFEA PROCESSOR
        ################################

        # Histogram Axes
        msoftdrop_axis = hist.axis.Regular(40, 0, 400, name="msoftdrop", label=r"Jet $m_\mathrm{softdrop}$ [GeV]")

        self.msoftdrop_axis = msoftdrop_axis
        self.n = 10  # Number of histograms
        self.make_output = lambda: self.create_histogram_matrix(self.n, self.msoftdrop_axis)

    def create_histogram_matrix(self, n, msoftdrop_axis):
        return [
            {  # Slot 0 for regular histograms
                f"b{i}": dah.Hist(
                    msoftdrop_axis,
                    storage=hist.storage.Weight()
                )
                for i in range(n)
            },
            {  # Slot 1 for z-based histograms
                f"b{i}": dah.Hist(
                    msoftdrop_axis,
                    storage=hist.storage.Weight()
                )
                for i in range(n)
            }
        ]

    def normalize(self, val, cut=None):
        if cut is None:
            return ak.fill_none(val, np.nan)
        else:
            return ak.fill_none(val[cut], np.nan)

    def process(self, events):
        output = self.make_output()

        ##################
        # OBJECT SELECTION
        ##################

        # For soft drop studies we care about the AK8 jets
        fatjets = events.FatJet
        fj_cut = (fatjets.pt > 450) & (abs(fatjets.eta) < 2.5)
        candidatejet = fatjets[fj_cut]

        leadingjets = candidatejet[:, 0:1]
        one_leading_jet = (ak.num(leadingjets, axis=1) == 1)
        leadingjets = leadingjets[one_leading_jet]

        select_events = (one_leading_jet) & (ak.any(fj_cut, axis=1))
        leadingjets = ak.flatten(leadingjets, axis=0)

        jetpt = ak.firsts(leadingjets.pt)
        jeteta = ak.firsts(leadingjets.eta)
        jetmsoftdrop = ak.firsts(leadingjets.msoftdrop)

        pf = ak.flatten(leadingjets.constituents.pf, axis=1)
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)
        cluster = fastjet.ClusterSequence(pf, jetdef)

        beta_softdrop_matrix = []
        z_softdrop_matrix = []
        calc_jetmsoftdrop = []
        z_calc_jetmsoftdrop = []

        for i in range(self.n):
            beta_softdrop_matrix.append(cluster.exclusive_jets_softdrop_grooming(beta=i / 10))
            z_softdrop_matrix.append(cluster.exclusive_jets_softdrop_grooming(symmetry_cut=i / 10))
            z_calc_jetmsoftdrop.append(ak.flatten(z_softdrop_matrix[i].msoftdrop, axis=0))
            calc_jetmsoftdrop.append(ak.flatten(beta_softdrop_matrix[i].msoftdrop, axis=0))

        ################
        # EVENT WEIGHTS
        ################
        weights = Weights(size=None, storeIndividual=True)
        output[0]["sumw"] = ak.sum(events.genWeight[select_events])
        weights.add("genweight", events.genWeight[select_events])

        ###################
        # FILL HISTOGRAMS
        ###################
        self.fill_histograms(output[0], calc_jetmsoftdrop, weights)
        self.fill_histograms(output[1], z_calc_jetmsoftdrop, weights)

        return output

    def fill_histograms(self, output, calc_jetmsoftdrop, weights):
        for i in range(self.n):
            output[f"b{i}"].fill(
                msoftdrop=self.normalize(calc_jetmsoftdrop[i]),
                weight=weights.weight()
            )

    def postprocess(self, accumulator):
        return accumulator
