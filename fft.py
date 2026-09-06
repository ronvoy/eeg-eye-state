import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLING_RATE = 128  # Hz, typical for EEG devices
INPUT_FILE = 'dataset/eeg_data.csv'
OUTPUT_FILE = 'eeg_freq.csv'

# Brain wave frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 100)
}

BAND_COLORS = {
    'delta': '#8B0000',
    'theta': '#FF4500',
    'alpha': '#FFD700',
    'beta': '#00CED1',
    'gamma': '#9370DB'
}

def load_data(filepath):
    """Load EEG data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples from {filepath}")
        print(f"✓ Channels: {', '.join([col for col in df.columns if col != 'eyeDetection'])}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None

def classify_brain_waves(freqs, psd, band_name):
    """Extract power for a specific frequency band."""
    low, high = FREQ_BANDS[band_name]
    mask = (freqs >= low) & (freqs < high)
    power = np.sum(psd[mask]) if np.any(mask) else 0
    return power

def compute_fft_analysis(data, channels):
    """Compute FFT and classify brain waves for all samples."""
    fft_results = []
    
    print("\n🔄 Processing FFT analysis...")
    
    for idx, row in data.iterrows():
        fft_row = {
            'sampleIndex': idx,
            'eyeDetection': row['eyeDetection'],
            'eyeState': 'open' if row['eyeDetection'] == 0 else 'closed'
        }
        
        # Process each channel
        for channel in channels:
            signal = np.array([row[channel]])
            
            # Compute FFT
            fft_vals = fft(signal)
            freqs = fftfreq(len(signal), 1/SAMPLING_RATE)
            psd = np.abs(fft_vals) ** 2
            
            # Classify into brain wave bands
            bands_power = {}
            for band_name in FREQ_BANDS.keys():
                power = classify_brain_waves(freqs, psd, band_name)
                bands_power[band_name] = power
                fft_row[f'{channel}_{band_name}'] = round(power, 6)
            
            # Find dominant band
            dominant_band = max(bands_power, key=bands_power.get)
            fft_row[f'{channel}_dominant'] = dominant_band
        
        fft_results.append(fft_row)
        
        if (idx + 1) % max(1, len(data) // 10) == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples")
    
    return pd.DataFrame(fft_results)

def save_fft_results(fft_df, filepath):
    """Save FFT results to CSV."""
    fft_df.to_csv(filepath, index=False)
    print(f"\n✓ FFT results saved to {filepath}")
    print(f"✓ Total columns: {len(fft_df.columns)}")

def create_frequency_spectrum_visualization(data, channels):
    """Create frequency spectrum visualization over time for all electrodes."""
    print("\n🔄 Creating frequency spectrum visualization...")
    
    # Number of channels
    num_channels = len(channels)
    
    # Create subplots for each channel
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 3 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    fig.suptitle('EEG Frequency Spectrum Over Time - All Electrodes', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, channel in enumerate(channels):
        # Extract signal for this channel
        signal = data[channel].values
        
        # Compute FFT
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), 1/SAMPLING_RATE)
        
        # Get positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        power = magnitude ** 2
        
        # Create spectrogram-like visualization
        ax = axes[idx]
        
        # Plot frequency spectrum
        ax.plot(positive_freqs, power, linewidth=1.5, color='#1f77b4', alpha=0.7)
        ax.fill_between(positive_freqs, power, alpha=0.3, color='#1f77b4')
        
        # Add frequency band markers
        for band, (low, high) in FREQ_BANDS.items():
            ax.axvspan(low, high, alpha=0.1, color=BAND_COLORS[band], label=band.capitalize())
            ax.text((low + high) / 2, ax.get_ylim()[1] * 0.9, band.upper(), 
                   ha='center', fontsize=9, fontweight='bold', alpha=0.7)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{channel} Power (μV²)', fontsize=11, fontweight='bold')
        ax.set_title(f'Electrode: {channel}', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)  # Focus on 0-60 Hz range
    
    plt.tight_layout()
    plt.savefig('eeg_frequency_spectrum.png', dpi=300, bbox_inches='tight')
    print("✓ Frequency spectrum visualization saved as 'eeg_frequency_spectrum.png'")
    plt.show()

def create_spectrogram_visualization(data, channels):
    """Create spectrogram visualization (frequency vs time) for all electrodes."""
    print("\n🔄 Creating spectrogram visualization...")
    
    num_channels = len(channels)
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 3 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    fig.suptitle('EEG Spectrogram (Frequency vs Time) - All Electrodes', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, channel in enumerate(channels):
        ax = axes[idx]
        
        # Extract signal for this channel
        signal = data[channel].values
        
        # Parameters for spectrogram
        window_size = 32  # Samples per window
        overlap = 16  # Overlap samples
        
        # Compute spectrogram manually
        freqs_list = []
        times_list = []
        spec_data = []
        
        for start in range(0, len(signal) - window_size, window_size - overlap):
            window = signal[start:start + window_size]
            
            # Apply Hann window
            windowed = window * np.hanning(len(window))
            
            # FFT
            fft_vals = fft(windowed)
            freqs = fftfreq(len(windowed), 1/SAMPLING_RATE)
            power = np.abs(fft_vals) ** 2
            
            # Positive frequencies only
            positive_idx = freqs > 0
            spec_data.append(power[positive_idx])
            freqs_list.append(freqs[positive_idx])
            times_list.append(start / SAMPLING_RATE)
        
        # Prepare data for spectrogram
        spec_array = np.array([np.interp(np.linspace(0, 60, 100), f, s) 
                               for f, s in zip(freqs_list, spec_data)])
        
        # Plot spectrogram
        im = ax.imshow(spec_array.T, aspect='auto', origin='lower', cmap='viridis',
                       extent=[times_list[0], times_list[-1], 0, 60],
                       interpolation='bilinear')
        
        ax.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'Electrode: {channel}', fontsize=12, fontweight='bold', loc='left')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Power (μV²)')
        
        # Add frequency band lines
        for band, (low, high) in FREQ_BANDS.items():
            ax.axhline(low, color=BAND_COLORS[band], linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(high, color=BAND_COLORS[band], linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('eeg_spectrogram.png', dpi=300, bbox_inches='tight')
    print("✓ Spectrogram visualization saved as 'eeg_spectrogram.png'")
    plt.show()

def create_brain_wave_timeline(data, fft_df, channels):
    """Create timeline visualization of dominant brain waves."""
    print("\n🔄 Creating brain wave timeline visualization...")
    
    num_channels = len(channels)
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    fig.suptitle('Dominant Brain Wave Timeline - All Electrodes', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, channel in enumerate(channels):
        ax = axes[idx]
        
        # Get dominant band for this channel
        dominant_col = f'{channel}_dominant'
        dominant_bands = fft_df[dominant_col].values
        
        # Map bands to numeric values for plotting
        band_to_num = {band: i for i, band in enumerate(FREQ_BANDS.keys())}
        band_nums = [band_to_num[b] for b in dominant_bands]
        
        # Create colored timeline
        for i in range(len(band_nums) - 1):
            band = dominant_bands[i]
            color = BAND_COLORS[band]
            ax.bar(i, 1, color=color, width=1, edgecolor='none')
        
        ax.set_ylabel(channel, fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f'Electrode: {channel}', fontsize=12, fontweight='bold', loc='left')
        
        # Add legend
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=BAND_COLORS[band], label=band.capitalize()) 
                              for band in FREQ_BANDS.keys()]
            ax.legend(handles=legend_elements, loc='upper right', ncol=5)
    
    plt.tight_layout()
    plt.savefig('eeg_brain_wave_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Brain wave timeline visualization saved as 'eeg_brain_wave_timeline.png'")
    plt.show()

def print_summary(data, fft_df):
    """Print analysis summary."""
    print("\n" + "="*60)
    print("📊 EEG FFT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total samples: {len(fft_df)}")
    print(f"  Eyes open: {len(fft_df[fft_df['eyeState'] == 'open'])}")
    print(f"  Eyes closed: {len(fft_df[fft_df['eyeState'] == 'closed'])}")
    
    print(f"\n🧠 Brain Wave Distribution:")
    dominant_cols = [col for col in fft_df.columns if 'dominant' in col]
    all_dominants = []
    for col in dominant_cols:
        all_dominants.extend(fft_df[col].values)
    
    for band in FREQ_BANDS.keys():
        count = all_dominants.count(band)
        percentage = (count / len(all_dominants)) * 100 if len(all_dominants) > 0 else 0
        print(f"  {band.capitalize():8s}: {count:4d} occurrences ({percentage:5.1f}%)")
    
    print(f"\n📋 Output Files:")
    print(f"  ✓ FFT Data: {OUTPUT_FILE}")
    print(f"  ✓ Frequency Spectrum: eeg_frequency_spectrum.png")
    print(f"  ✓ Spectrogram: eeg_spectrogram.png")
    print(f"  ✓ Brain Wave Timeline: eeg_brain_wave_timeline.png")
    print("\n" + "="*60)

def main():
    print("\n🧠 EEG FFT Brain Wave Analyzer")
    print("="*60)
    
    # Load data
    data = load_data(INPUT_FILE)
    if data is None:
        return
    
    # Get channel names
    channels = [col for col in data.columns if col != 'eyeDetection']
    
    # Perform FFT analysis
    fft_df = compute_fft_analysis(data, channels)
    
    # Save results
    save_fft_results(fft_df, OUTPUT_FILE)
    
    # Create visualizations
    create_frequency_spectrum_visualization(data, channels)
    create_spectrogram_visualization(data, channels)
    create_brain_wave_timeline(fft_df, fft_df, channels)
    
    # Print summary
    print_summary(data, fft_df)

if __name__ == '__main__':
    main()