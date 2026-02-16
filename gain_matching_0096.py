import ROOT
import matplotlib.pyplot as plt
import numpy as np
import csv

energy = [2700, 5190, 5490]
mean = []
meanErr = []
xmin = 0
xmax = 8000
file_path = "/Users/kjswaff/Documents/Research/mcc_analysis/run-0096-nocal.root"
f = ROOT.TFile.Open(file_path, "READ")
folder = f.Get("Gamma;1")
if not folder:
    raise RuntimeError("Folder not found")
f.cd("Gamma;1")

peak_data = []
successful_detectors = []

for i in range(52):
    h_name = f"hge_eraw_{i};1"
    h = ROOT.gDirectory.Get(h_name)
    if not h:
        raise RuntimeError(f"Histogram {h_name} not found")
    if h.GetEntries() < 10:
        continue
    
    mean_i = []
    mean_i_Err = []
    
    spectrum = ROOT.TSpectrum()
    lower = 2300
    upper = 6000
    h.GetXaxis().SetRangeUser(lower, upper)
    n_peaks = spectrum.Search(h, 2, "", 0.004)
    x_pos = spectrum.GetPositionX()
    
    # change based on where you expect peaks
    lower_1 = 2500
    upper_1 = 2900
    lower_2 = 4500
    upper_2 = 6000
    
    peaks = []
    for j in range(n_peaks):
        peak_x = x_pos[j]
        if (lower_1 <= peak_x <= upper_1) or (lower_2 <= peak_x <= upper_2):
            peaks.append(peak_x)
    
    peaks.sort()
    
    if len(peaks) != 3:
        print(f"Warning: Detector {i} found {len(peaks)} peaks instead of 3. Peaks at: {peaks}")
        continue
    
    h.GetXaxis().SetRangeUser(xmin, xmax)
    c = ROOT.TCanvas()
    h.Draw()
    c.SetLogy(True)
    #c.SaveAs(f"Detector_{i}.png")
    
    for j in range(3):
        peak_pos = peaks[j]
        gaussFit = ROOT.TF1("gaussfit", "gaus", peak_pos - 20, peak_pos + 20)
        h.GetXaxis().SetRangeUser(peak_pos - 20, peak_pos + 20)
        h.Fit(gaussFit, "EQ")  # Added Q for quiet mode
        
        mean_j = gaussFit.GetParameter(1)
        mean_i.append(mean_j)
        meanErr_j = gaussFit.GetParError(1)
        mean_i_Err.append(meanErr_j)
        
        peak_data.append({
            'detector': i,
            'peak_number': j,
            'initial_peak_position': peak_pos,
            'fitted_mean': mean_j,
            'fitted_mean_error': meanErr_j,
            'expected_channel': energy[j]
        })
        
        h.Draw()
        c.SetLogy(True)
        #c.SaveAs(f"Detector_{i}_Peak_{j}.png")
    
    mean.append(mean_i)
    meanErr.append(mean_i_Err)
    successful_detectors.append(i)
c.Close()

for i in range(52):
    if i not in successful_detectors:
        for j in range(3):
            peak_data.append({
                'detector': i,
                'peak_number': j,
                'initial_peak_position': 0,
                'fitted_mean': 0,
                'fitted_mean_error': 0,
                'expected_channel': energy[j]
            })

peak_data.sort(key=lambda x: (x['detector'], x['peak_number']))

with open('peak_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['detector', 'peak_number', 'initial_peak_position', 'fitted_mean', 'fitted_mean_error', 'expected_channel']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in peak_data:
        writer.writerow(row)

print("Peak data saved to peak_data.csv")

gain_matching_data = []

if len(mean) < 2:
    print("Not enough detectors with valid data for gain matching")
    for i in range(1, 52):
        gain_matching_data.append({
            'detector_pair': f"0_vs_{i}",
            'slope': 0,
            'intercept': 0
        })
else:
    x = np.array(mean[0])
    detector_mean_map = {successful_detectors[idx]: mean[idx] for idx in range(len(successful_detectors))}
    
    for i in range(1, 52):
        if i in successful_detectors:
            idx = successful_detectors.index(i)
            y = np.array(mean[idx])
            m, b = np.polyfit(x, y, 1)
            y_fit = m * x + b
            
            gain_matching_data.append({
                'detector_pair': f"0_vs_{i}",
                'slope': m,
                'intercept': b
            })
            
            plt.figure()
            plt.scatter(x, y, label="Detector_0")
            plt.plot(x, y_fit, color="red", label=f"Fit: y = {m:.2f}x + {b:.2f}")
            plt.title(f"Detector_0 vs Detector_{i}")
            plt.xlabel("Detector_0")
            plt.ylabel(f"Detector_{i}")
            plt.legend()
            #plt.savefig(f"Detector_0_vs_Detector_{i}.png")
            plt.close()
        else:
            gain_matching_data.append({
                'detector_pair': f"0_vs_{i}",
                'slope': 0,
                'intercept': 0
            })

with open('gain_matching_parameters.csv', 'w', newline='') as csvfile:
    fieldnames = ['detector_pair', 'slope', 'intercept']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in gain_matching_data:
        writer.writerow(row)

print("Gain matching parameters saved to gain_matching_parameters.csv")

gain_params = {}
with open('gain_matching_parameters.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        detector_pair = row['detector_pair']
        detector_num = int(detector_pair.split('_vs_')[1])
        gain_params[detector_num] = {
            'slope': float(row['slope']),
            'intercept': float(row['intercept'])
        }

print(f"Loaded gain matching parameters for {len(gain_params)} detectors")

print("\nFirst 10 detector parameters:")
for i in range(10):
    if i in gain_params:
        print(f"  Detector {i}: slope={gain_params[i]['slope']:.6f}, intercept={gain_params[i]['intercept']:.6f}")
    else:
        print(f"  Detector {i}: NOT FOUND IN CSV")
print()

h_0 = folder.Get("hge_eraw_0;1")
if not h_0:
    raise RuntimeError("Could not load Detector 0 histogram!")

output_file = ROOT.TFile.Open("run-0096_Gain_Matched.root", "RECREATE")
output_folder = output_file.mkdir("Gamma")
output_folder.cd()

for i in range(52):
    print(f"Processing Detector {i}...")
    
    if i == 0:
        # Detector 0 is the reference - just clone it
        hist_corrected = h_0.Clone(f"hge_eraw_gain_matched_{i}")
        hist_corrected.SetTitle(f"Gain-Matched Detector {i} (Reference)")
        hist_corrected.Write()
        print(f"  Detector {i}: Reference detector (no correction needed)")
        continue
    
    params = gain_params.get(i, {'slope': 0, 'intercept': 0})
    m_fit = params['slope']
    b_fit = params['intercept']
    
    # check if detector was skipped, slope should be 0
    if m_fit == 0:
        xaxis_ref = h_0.GetXaxis()
        nbins = h_0.GetNbinsX()
        xmin = xaxis_ref.GetXmin()
        xmax = xaxis_ref.GetXmax()
        
        hist_corrected = ROOT.TH1F(f"hge_eraw_gain_matched_{i}", 
                                    f"Gain-Matched Detector {i} (Skipped)", 
                                    nbins, xmin, xmax)
        hist_corrected.Write()
        print(f"  Detector {i}: Skipped (empty histogram created)")
        continue
    
    h_name = f"hge_eraw_{i};1"
    h_i = folder.Get(h_name)
    
    if not h_i:
        print(f"  Warning: Could not load histogram for Detector {i}, creating empty histogram")
        print(f"    Tried to get: {h_name}")
        xaxis_ref = h_0.GetXaxis()
        nbins = h_0.GetNbinsX()
        xmin = xaxis_ref.GetXmin()
        xmax = xaxis_ref.GetXmax()
        
        hist_corrected = ROOT.TH1F(f"hge_eraw_gain_matched_{i}", 
                                    f"Gain-Matched Detector {i} (Missing)", 
                                    nbins, xmin, xmax)
        hist_corrected.Write()
        continue
    
    print(f"  Successfully loaded histogram for Detector {i}")
    print(f"    Entries: {h_i.GetEntries()}")
    
    gain_correction = 1.0 / m_fit
    offset_correction = -b_fit / m_fit
    
    hist_corrected = h_i.Clone(f"hge_eraw_gain_matched_{i}")
    hist_corrected.SetTitle(f"Gain-Matched Detector {i}")
    
    xaxis = hist_corrected.GetXaxis()
    nbins = hist_corrected.GetNbinsX()
    
    new_edges = []
    for j in range(nbins + 1):  # +1 because we need nbins+1 edges
        old_edge = xaxis.GetBinLowEdge(j+1) if j < nbins else xaxis.GetBinUpEdge(nbins)
        new_edge = gain_correction * old_edge + offset_correction
        new_edges.append(new_edge)
    
    xaxis.Set(nbins, np.array(new_edges))
    
    hist_corrected.Write()
    print(" Detector {i}: Gain matched (slope={m_fit:.4f}, intercept={b_fit:.4f})")

output_file.Close()
f.Close()

print("\n" + "="*60)
print("Gain matching complete!")
print(f"Output saved to: run-0096_Gain_Matched.root")
print(f"Folder name: GainMatched")
print("="*60)

print("\nCreating comparison plots for all detectors...")

f = ROOT.TFile.Open(file_path, "READ")
input_folder_plot = f.Get("Gamma;1")
h_0 = input_folder_plot.Get("hge_eraw_0;1")

output_file = ROOT.TFile.Open("run-0096_Gain_Matched.root", "READ")
output_folder = output_file.Get("Gamma")

for i in range(1, 52):
    params = gain_params.get(i, {'slope': 0, 'intercept': 0})
    m_fit = params['slope']
    b_fit = params['intercept']
    
    hist_corrected = output_folder.Get(f"hge_eraw_gain_matched_{i}")
    if not hist_corrected:
        print(f"Warning: Could not load gain-matched histogram for Detector {i}")
        continue

    h_name = f"hge_eraw_{i};1"
    h_i = input_folder_plot.Get(h_name)
    
    is_skipped = (m_fit == 0)
    is_missing = (h_i is None or h_i.GetEntries() == 0)
    
    canvas = ROOT.TCanvas(f"canvas_{i}", f"Gain Matching Comparison - Detector {i}", 1400, 700)
    canvas.Divide(2, 1)
    
    canvas.cd(1)
    ROOT.gPad.SetLogy(True)
    
    if is_skipped or is_missing:
        dummy = h_0.Clone("dummy")
        dummy.Reset()
        dummy.SetTitle(f"Before Gain Matching - Detector {i}")
        dummy.GetXaxis().SetRangeUser(2000, 6000)
        dummy.GetXaxis().SetTitle("Energy [ADC]")
        dummy.GetYaxis().SetTitle("Counts")
        dummy.SetStats(0)
        dummy.Draw("HIST")
        
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextAlign(22)
        text.SetTextSize(0.05)
        text.SetTextColor(ROOT.kRed)
        if is_skipped:
            text.DrawLatex(0.5, 0.5, "Detector Skipped")
            text.SetTextSize(0.03)
            text.DrawLatex(0.5, 0.45, "(No valid peaks found)")
        else:
            text.DrawLatex(0.5, 0.5, "No Data Available")
            text.SetTextSize(0.03)
            text.DrawLatex(0.5, 0.45, "(Histogram missing or empty)")
        
        h_0.SetLineColor(ROOT.kBlue)
        h_0.SetLineWidth(2)
        h_0.SetFillStyle(0)
        h_0.Draw("HIST SAME")
        
        legend1 = ROOT.TLegend(0.35, 0.75, 0.90, 0.87)
        legend1.AddEntry(h_0, "Detector 0 (reference, blue)", "l")
        legend1.SetTextSize(0.035)
        legend1.SetBorderSize(0)
        legend1.Draw()
    else:
        h_0.SetLineColor(ROOT.kBlue)
        h_0.SetLineWidth(2)
        h_0.SetFillStyle(0)
        
        h_i.SetLineColor(ROOT.kGreen + 2)
        h_i.SetLineWidth(2)
        h_i.SetFillStyle(0)
        
        h_i.SetTitle(f"Before Gain Matching - Detector {i}")
        h_i.GetXaxis().SetRangeUser(2000, 6000)
        h_i.GetXaxis().SetTitle("Energy [ADC]")
        h_i.GetYaxis().SetTitle("Counts")
        h_i.SetStats(0)
        
        h_i.Draw("HIST")
        h_0.Draw("HIST SAME")
        
        legend1 = ROOT.TLegend(0.35, 0.75, 0.90, 0.87)
        legend1.AddEntry(h_0, "Detector 0 (reference, blue)", "l")
        legend1.AddEntry(h_i, f"Detector {i} (raw, green)", "l")
        legend1.SetTextSize(0.035)
        legend1.SetBorderSize(0)
        legend1.Draw()
    
    canvas.cd(2)
    ROOT.gPad.SetLogy(True)
    
    if is_skipped or is_missing:
        dummy2 = h_0.Clone("dummy2")
        dummy2.Reset()
        dummy2.SetTitle(f"After Gain Matching - Detector {i}")
        dummy2.GetXaxis().SetRangeUser(2000, 6000)
        dummy2.GetXaxis().SetTitle("Energy [ADC]")
        dummy2.GetYaxis().SetTitle("Counts")
        dummy2.SetStats(0)
        dummy2.Draw("HIST")
        
        text2 = ROOT.TLatex()
        text2.SetNDC()
        text2.SetTextAlign(22)
        text2.SetTextSize(0.05)
        text2.SetTextColor(ROOT.kRed)
        if is_skipped:
            text2.DrawLatex(0.5, 0.5, "No Correction Applied")
            text2.SetTextSize(0.03)
            text2.DrawLatex(0.5, 0.45, "(Detector was skipped)")
        else:
            text2.DrawLatex(0.5, 0.5, "No Correction Applied")
            text2.SetTextSize(0.03)
            text2.DrawLatex(0.5, 0.45, "(No data available)")
        
        h_0.Draw("HIST SAME")
        
        legend2 = ROOT.TLegend(0.35, 0.75, 0.90, 0.87)
        legend2.AddEntry(h_0, "Detector 0 (reference, blue)", "l")
        legend2.SetTextSize(0.035)
        legend2.SetBorderSize(0)
        legend2.Draw()
    else:
        hist_corrected.SetLineColor(ROOT.kRed)
        hist_corrected.SetLineWidth(2)
        hist_corrected.SetFillStyle(0)
        
        h_0.SetTitle(f"After Gain Matching - Detector {i}")
        h_0.GetXaxis().SetRangeUser(2000, 6000)
        h_0.GetXaxis().SetTitle("Energy [ADC]")
        h_0.GetYaxis().SetTitle("Counts")
        h_0.SetStats(0)
        
        h_0.Draw("HIST")
        hist_corrected.Draw("HIST SAME")
        
        legend2 = ROOT.TLegend(0.35, 0.75, 0.90, 0.87)
        legend2.AddEntry(h_0, "Detector 0 (reference, blue)", "l")
        legend2.AddEntry(hist_corrected, f"Detector {i} (corrected, red)", "l")
        legend2.SetTextSize(0.035)
        legend2.SetBorderSize(0)
        legend2.Draw()
    
    canvas.Update()
    canvas.SaveAs(f"Detector_{i}_gain_match_comparison.png")
    
    if is_skipped:
        print(f"Saved comparison plot: Detector_{i}_gain_match_comparison.png (SKIPPED)")
    elif is_missing:
        print(f"Saved comparison plot: Detector_{i}_gain_match_comparison.png (NO DATA)")
    else:
        print(f"Saved comparison plot: Detector_{i}_gain_match_comparison.png (slope={m_fit:.4f}, intercept={b_fit:.4f})")

output_file.Close()
f.Close()

print("\nAll done!")