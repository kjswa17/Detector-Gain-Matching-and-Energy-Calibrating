import ROOT
import matplotlib.pyplot as plt
import numpy as np
import csv

# calibration energies
energy = [511, 984, 1460, 2614]

# expected windows, update appropriately
peak_windows = {
    511:  (2600, 2800),
    984:  (5100, 5250),
    1460: (7600, 7800),
    2614: (13600, 13800),  
}

# gaussian fit half-width
fit_half_width = 15

file_path = "/Users/kjswaff/Documents/Research/mcc_analysis/run-0096_gain_matching/run-0096_Gain_Matched.root"
f = ROOT.TFile.Open(file_path, "READ")
folder = f.Get("Gamma;1")
if not folder:
    raise RuntimeError("Folder not found")
f.cd("Gamma;1")

peak_data = []
successful_detectors = []
all_fitted_channels = []  # Store fitted channel positions for each detector

print("="*60)
print("4-POINT LINEAR CALIBRATION")
print("Fit: Energy = m * channel + b")
print(f"Energies: {energy} keV")
print("="*60)

for i in range(52):
    h_name = f"hge_eraw_gain_matched_{i};1"
    h = ROOT.gDirectory.Get(h_name)
    if not h:
        continue
    if h.GetEntries() < 10:
        continue
    
    fitted_channels = []
    fitted_sigmas = []
    peaks_found = []
    
    found_all = True
    
    # Create figure for Gaussian fits
    fig_gauss, axes_gauss = plt.subplots(2, 3, figsize=(15, 10))
    axes_gauss = axes_gauss.flatten()
    
    for idx, e in enumerate(energy):
        lower, upper = peak_windows[e]
        
        spectrum = ROOT.TSpectrum()
        h.GetXaxis().SetRangeUser(lower, upper)
        n_peaks = spectrum.Search(h, 2, "", 0.01)
        
        if n_peaks == 0:
            print(f"Detector {i}: No peak found for {e} keV in [{lower}, {upper}]")
            found_all = False
            break
        
        x_pos = spectrum.GetPositionX()
        y_pos = spectrum.GetPositionY()
        
        # Find highest amplitude peak
        best_idx = 0
        for j in range(n_peaks):
            if y_pos[j] > y_pos[best_idx]:
                best_idx = j
        
        peak_pos = x_pos[best_idx]
        peaks_found.append(peak_pos)
        
        # perform Gaussian fit
        fit_low = peak_pos - fit_half_width
        fit_high = peak_pos + fit_half_width
        
        gaussFit = ROOT.TF1(f"gaus_{i}_{e}", "gaus", fit_low, fit_high)
        bin_at_peak = h.FindBin(peak_pos)
        amp_estimate = h.GetBinContent(bin_at_peak)
        
        gaussFit.SetParameter(0, amp_estimate)
        gaussFit.SetParameter(1, peak_pos)
        gaussFit.SetParameter(2, 5)
        
        h.GetXaxis().SetRangeUser(fit_low, fit_high)
        h.Fit(gaussFit, "RQ")
        
        fitted_mean = gaussFit.GetParameter(1)
        fitted_sigma = gaussFit.GetParameter(2)
        fitted_amp = gaussFit.GetParameter(0)
        
        fitted_channels.append(fitted_mean)
        fitted_sigmas.append(fitted_sigma)
        
        peak_data.append({
            'detector': i,
            'energy': e,
            'tspectrum_pos': peak_pos,
            'fitted_mean': fitted_mean,
            'fitted_sigma': fitted_sigma
        })
        
        # Plot Gaussian fit
        ax = axes_gauss[idx]
        
        # Extract data for plotting
        plot_low = peak_pos - 50
        plot_high = peak_pos + 50
        h.GetXaxis().SetRangeUser(plot_low, plot_high)
        
        bins_x = []
        bins_y = []
        for b in range(1, h.GetNbinsX() + 1):
            bc = h.GetBinCenter(b)
            if plot_low <= bc <= plot_high:
                bins_x.append(bc)
                bins_y.append(h.GetBinContent(b))
        
        ax.step(bins_x, bins_y, 'b-', where='mid', linewidth=1, label='Data')
        
        # Plot Gaussian fit
        x_fit = np.linspace(fit_low, fit_high, 100)
        y_fit = fitted_amp * np.exp(-0.5 * ((x_fit - fitted_mean) / fitted_sigma)**2)
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Gaussian fit')
        
        # Mark fit range
        ax.axvline(fit_low, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(fit_high, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Counts')
        ax.set_title(f'{e} keV\nμ={fitted_mean:.1f}, σ={fitted_sigma:.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    h.GetXaxis().SetRangeUser(0, 16000)
    
    if not found_all or len(fitted_channels) != 4:
        plt.close(fig_gauss)
        peak_data = [p for p in peak_data if p['detector'] != i]
        continue
    
    ax_full = axes_gauss[4]
    h.GetXaxis().SetRangeUser(2000, 15000)
    
    bins_x_full = []
    bins_y_full = []
    for b in range(1, h.GetNbinsX() + 1):
        bc = h.GetBinCenter(b)
        if 2000 <= bc <= 15000:
            bins_x_full.append(bc)
            bins_y_full.append(h.GetBinContent(b))
    
    ax_full.semilogy(bins_x_full, bins_y_full, 'b-', linewidth=0.5)
    for idx, (e, ch) in enumerate(zip(energy, fitted_channels)):
        ax_full.axvline(ch, color='red', linestyle='--', alpha=0.7)
        ax_full.text(ch, ax_full.get_ylim()[1]*0.5, f'{e}', rotation=90, 
                    va='top', ha='right', fontsize=8, color='red')
    ax_full.set_xlabel('Channel')
    ax_full.set_ylabel('Counts')
    ax_full.set_title('Full Spectrum with Fitted Peaks')
    ax_full.grid(True, alpha=0.3)
    
    axes_gauss[5].axis('off')
    
    fig_gauss.suptitle(f'Detector {i} - Gaussian Fits', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"Detector_{i}_Gaussian_Fits.png", dpi=150)
    plt.close(fig_gauss)
    
    all_fitted_channels.append(fitted_channels)
    successful_detectors.append(i)
    print(f"Detector {i}: Fitted channels = {[f'{ch:.1f}' for ch in fitted_channels]}")

print(f"\nSuccessfully processed {len(successful_detectors)}/52 detectors")

if len(successful_detectors) == 0:
    print("ERROR: No detectors processed!")
    f.Close()
    exit()

# perform linear: Energy = m * channel + b
# x = channel, y = energy
calibration_data = []

print("\n" + "="*60)
print("LINEAR CALIBRATION: Energy = m * channel + b")
print("="*60)

for idx, detector_num in enumerate(successful_detectors):
    x = np.array(all_fitted_channels[idx])  # channels
    y = np.array(energy)  # energies
    
    # Linear fit: E = m * ch + b
    m, b = np.polyfit(x, y, 1)
    
    y_fit = m * x + b
    residuals = y - y_fit
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    calibration_data.append({
        'detector': detector_num,
        'slope': m,
        'intercept': b,
        'r_squared': r_squared
    })
    
    print(f"\nDetector {detector_num}:")
    print(f"  E = {m:.6f} * channel + {b:.4f}")
    print(f"  R² = {r_squared:.8f}")
    print(f"  Residuals (keV): {[f'{r:+.2f}' for r in residuals]}")
    
    # Plot calibration curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
    
    x_smooth = np.linspace(min(x) - 200, max(x) + 200, 200)
    y_smooth = m * x_smooth + b
    
    ax1.scatter(x, y, s=100, zorder=5, label="Calibration Points")
    ax1.plot(x_smooth, y_smooth, 'r-', label=f'E = {m:.6f}×ch + {b:.2f}\nR² = {r_squared:.8f}')
    for xi, yi, res in zip(x, y, residuals):
        ax1.annotate(f'{yi} keV\nres={res:+.1f}', (xi, yi), 
                    textcoords="offset points", xytext=(5, 10), ha='left', fontsize=8)
    ax1.set_title(f"Energy Calibration - Detector {detector_num}")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Energy [keV]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x, residuals, width=50, color='green', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel("Channel")
    ax2.set_ylabel("Residual [keV]")
    ax2.set_title(f"Residuals (max = {np.max(np.abs(residuals)):.2f} keV)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"Energy_vs_Detector_{detector_num}.png", dpi=150)
    plt.close()

# Save calibration parameters
with open('calibration_parameters.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['detector', 'slope', 'intercept', 'r_squared'])
    writer.writeheader()
    writer.writerows(calibration_data)

print("\nCalibration parameters saved to calibration_parameters.csv")

calibration_params = {row['detector']: {'slope': row['slope'], 'intercept': row['intercept']} 
                      for row in calibration_data}

f.Close()

# Create calibrated histograms
print("\n" + "="*60)
print("Creating calibrated histograms...")
print("Using bin-edge rescaling method (same as gain matching)")
print("="*60)

f = ROOT.TFile.Open(file_path, "READ")
folder = f.Get("Gamma;1")

output_file = ROOT.TFile.Open("run-0096_Calibrated.root", "RECREATE")
output_folder = output_file.mkdir("Gamma")
output_folder.cd()

for i in range(52):
    h_i = folder.Get(f"hge_eraw_gain_matched_{i};1")
    
    if i not in calibration_params:
        # Create empty histogram for skipped detectors
        if h_i:
            hist_calibrated = h_i.Clone(f"hge_eraw_calibrated_{i}")
            hist_calibrated.Reset()
        else:
            hist_calibrated = ROOT.TH1F(f"hge_eraw_calibrated_{i}",
                                         f"Detector {i} (Skipped)",
                                         16000, 0, 16000)
        hist_calibrated.SetTitle(f"Detector {i} (Skipped)")
        hist_calibrated.GetXaxis().SetTitle("Energy [keV]")
        hist_calibrated.GetYaxis().SetTitle("Counts")
        ROOT.SetOwnership(hist_calibrated, False)
        hist_calibrated.Write()
        print(f"Detector {i}: Skipped (no calibration parameters)")
        continue
    
    if not h_i:
        hist_calibrated = ROOT.TH1F(f"hge_eraw_calibrated_{i}",
                                     f"Detector {i} (Missing)",
                                     16000, 0, 16000)
        hist_calibrated.GetXaxis().SetTitle("Energy [keV]")
        hist_calibrated.GetYaxis().SetTitle("Counts")
        ROOT.SetOwnership(hist_calibrated, False)
        hist_calibrated.Write()
        print(f"Detector {i}: Missing histogram")
        continue
    
    params = calibration_params[i]
    m = params['slope']
    b = params['intercept']
    
    # Clone the histogram and rescale bin edges 
    hist_calibrated = h_i.Clone(f"hge_eraw_calibrated_{i}")
    hist_calibrated.SetTitle(f"Energy-Calibrated Detector {i}")
    hist_calibrated.GetXaxis().SetTitle("Energy [keV]")
    hist_calibrated.GetYaxis().SetTitle("Counts")
    ROOT.SetOwnership(hist_calibrated, False)
    
    xaxis = hist_calibrated.GetXaxis()
    nbins = hist_calibrated.GetNbinsX()
    
    # Rescale bin edges: E = m * channel + b
    new_edges = []
    for j in range(nbins + 1):  # +1 because we need nbins+1 edges
        old_edge = xaxis.GetBinLowEdge(j+1) if j < nbins else xaxis.GetBinUpEdge(nbins)
        new_edge = m * old_edge + b
        new_edges.append(new_edge)
    
    xaxis.Set(nbins, np.array(new_edges))
    
    hist_calibrated.Write()
    print(f"Detector {i}: Calibrated (E = {m:.6f} * ch + {b:.4f})")

output_file.Close()
f.Close()

# Create summed spectrum
print("\n" + "="*60)
print("Creating summed spectrum...")
print("="*60)

ENERGY_MIN = 0
ENERGY_MAX = 4000
ENERGY_NBINS = 4000

output_file = ROOT.TFile.Open("run-0096_Calibrated.root", "UPDATE")
output_folder = output_file.Get("Gamma")

h_sum = ROOT.TH1F("hge_eraw_calibrated_sum", "Summed Energy-Calibrated Spectrum",
                   ENERGY_NBINS, ENERGY_MIN, ENERGY_MAX)
h_sum.GetXaxis().SetTitle("Energy [keV]")
h_sum.GetYaxis().SetTitle("Counts")
ROOT.SetOwnership(h_sum, False)

detectors_summed = []
sum_bin_width = (ENERGY_MAX - ENERGY_MIN) / float(ENERGY_NBINS)  # 1.0 keV per bin

for i in successful_detectors:
    h_i = output_folder.Get(f"hge_eraw_calibrated_{i}")
    if h_i and h_i.GetEntries() > 0:
        nbins_i = h_i.GetNbinsX()
        xaxis_i = h_i.GetXaxis()
        for bin_idx in range(1, nbins_i + 1):
            content = h_i.GetBinContent(bin_idx)
            if content == 0:
                continue

            src_low = xaxis_i.GetBinLowEdge(bin_idx)
            src_high = xaxis_i.GetBinUpEdge(bin_idx)
            src_width = src_high - src_low
            
            if src_width <= 0 or src_high <= ENERGY_MIN or src_low >= ENERGY_MAX:
                continue
            
            first_uniform = max(1, h_sum.FindBin(src_low))
            last_uniform = min(ENERGY_NBINS, h_sum.FindBin(src_high))
            
            for ub in range(first_uniform, last_uniform + 1):
                ub_low = h_sum.GetBinLowEdge(ub)
                ub_high = ub_low + sum_bin_width

                overlap_low = max(src_low, ub_low)
                overlap_high = min(src_high, ub_high)
                overlap = overlap_high - overlap_low
                
                if overlap > 0:
                    fraction = overlap / src_width
                    h_sum.SetBinContent(ub, h_sum.GetBinContent(ub) + content * fraction)
        
        detectors_summed.append(i)

print(f"Summed {len(detectors_summed)} detectors")
print(f"Total entries: {h_sum.GetEntries():.0f}")

output_folder.cd()
h_sum.Write()

n_bins = h_sum.GetNbinsX()
x_centers = np.array([h_sum.GetBinCenter(b) for b in range(1, n_bins + 1)])
y_values = np.array([h_sum.GetBinContent(b) for b in range(1, n_bins + 1)])

# Create zoomed peak plots 
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Full spectrum
ax_full = axes[0]
mask_full = (x_centers >= 0) & (x_centers <= 3000)
ax_full.semilogy(x_centers[mask_full], y_values[mask_full], 'k-', linewidth=0.8)
ax_full.set_xlabel('Energy (keV)')
ax_full.set_ylabel('Counts')
ax_full.set_title('Full Spectrum (4-point Linear Cal)')
for e in energy:
    ax_full.axvline(e, color='red', linestyle='--', alpha=0.5)
ax_full.grid(True, alpha=0.3)

# Zoomed views
zoom_configs = [
    (511, '511 keV', 30),
    (984, '984 keV', 30),
    (1460, '1460 keV', 30),
    (2614, '2614 keV', 30),
]

ax_list = [axes[1], axes[2], axes[3], axes[4]]

for ax, (e_center, label, half_range) in zip(ax_list, zoom_configs):
    mask = (x_centers >= e_center - half_range) & (x_centers <= e_center + half_range)
    if np.any(mask) and np.any(y_values[mask] > 0):
        ax.plot(x_centers[mask], y_values[mask], 'k-', linewidth=1)
        ax.axvline(e_center, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.set_title(f'{label}')
        ax.set_xlim(e_center - half_range, e_center + half_range)
        
        peak_idx = np.argmax(y_values[mask])
        actual_peak = x_centers[mask][peak_idx]
        offset = actual_peak - e_center
        
        color = 'lightgreen' if abs(offset) < 5 else 'yellow'
        ax.annotate(f'Peak: {actual_peak:.1f} keV\nOffset: {offset:+.1f} keV', 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        ax.grid(True, alpha=0.3)

axes[5].axis('off')

plt.tight_layout()
plt.savefig("Calibrated_Sum_Spectrum_ZoomedPeaks.png", dpi=150)
plt.close()
print("Saved: Calibrated_Sum_Spectrum_ZoomedPeaks.png")

# ROOT plots
canvas_sum = ROOT.TCanvas("canvas_sum", "Summed Spectrum", 1200, 800)
canvas_sum.SetLogy(True)
h_sum.SetLineColor(ROOT.kBlack)
h_sum.SetLineWidth(1)
h_sum.GetXaxis().SetRangeUser(0, 5000)
h_sum.SetStats(0)
h_sum.Draw("HIST")

lines_to_keep = []
for e in energy:
    line = ROOT.TLine(e, 1, e, h_sum.GetMaximum()*0.3)
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)
    line.Draw("SAME")
    lines_to_keep.append(line)

canvas_sum.Update()
canvas_sum.SaveAs("Calibrated_Sum_Spectrum.png")
canvas_sum.Close()
print("Saved: Calibrated_Sum_Spectrum.png")

output_file.Close()

print("\n" + "="*60)
print("4-POINT LINEAR CALIBRATION COMPLETE")
print(f"Calibration: Energy = m * channel + b")
print(f"Energies used: {energy} keV")
print(f"Calibrated {len(detectors_summed)}/52 detectors")
print("="*60)
print("\nOutput files:")
print("  - Detector_*_Gaussian_Fits.png (Gaussian fit diagnostics)")
print("  - Energy_vs_Detector_*.png (calibration curves)")
print("  - Calibrated_Sum_Spectrum_ZoomedPeaks.png")
print("  - Calibrated_Sum_Spectrum.png")
print("  - calibration_parameters.csv")
print("  - run-0096_Calibrated.root")