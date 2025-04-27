from IPython.display import display, Markdown
import matplotlib.pyplot as plt
from scipy.signal import welch

window_size = 5

def interpret_window(segment_upper, segment_lower, Pxx_upper, Pxx_lower, label, window_idx):
    interp = f"""
**Window {window_idx}: {label}**

- **Time Domain (Upper):** Range: {segment_upper.min():.2f} to {segment_upper.max():.2f}
- **Time Domain (Lower):** Range: {segment_lower.min():.2f} to {segment_lower.max():.2f}
- **PSD (Upper):** Max Power: {Pxx_upper.max():.2f}
- **PSD (Lower):** Max Power: {Pxx_lower.max():.2f}

- If the upper/lower load cell shows higher variance or PSD, it may indicate more dynamic movement in that window.
- Compare the difference between upper and lower to see if the load is distributed differently during movement.
"""
    display(Markdown(interp))

def sliding_window_plot(df, col, col2, label):
    for start in range(0, len(df) - window_size + 1):
        segment_upper = df[col].iloc[start:start+window_size].values
        segment_lower = df[col2].iloc[start:start+window_size].values

        # PSD
        f_upper, Pxx_upper = welch(segment_upper, nperseg=window_size)
        f_lower, Pxx_lower = welch(segment_lower, nperseg=window_size)

        # Stack plots: 2 rows (upper/lower), 2 cols (time/PSD)
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle(f'{label} - Window {start} to {start+window_size-1}')

        # Time domain
        axs[0, 0].plot(segment_upper, color='b')
        axs[0, 0].set_title('Upper Load Cell - Time')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].grid(True)

        axs[1, 0].plot(segment_lower, color='g')
        axs[1, 0].set_title('Lower Load Cell - Time')
        axs[1, 0].set_xlabel('Sample')
        axs[1, 0].set_ylabel('Value')
        axs[1, 0].grid(True)

        # PSD
        axs[0, 1].semilogy(f_upper, Pxx_upper, color='b')
        axs[0, 1].set_title('Upper Load Cell - PSD')
        axs[0, 1].set_xlabel('Frequency [Hz]')
        axs[0, 1].set_ylabel('PSD')
        axs[0, 1].grid(True)

        axs[1, 1].semilogy(f_lower, Pxx_lower, color='g')
        axs[1, 1].set_title('Lower Load Cell - PSD')
        axs[1, 1].set_xlabel('Frequency [Hz]')
        axs[1, 1].set_ylabel('PSD')
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Interpretation for this window
        interpret_window(segment_upper, segment_lower, Pxx_upper, Pxx_lower, label, start)

# Example usage for all three movement types:
# sliding_window_plot(df_no, col, col2, "No Movement")
# sliding_window_plot(df_1, col, col2, "Movement 1")
# sliding_window_plot(df_2, col, col2, "Movement 2")
