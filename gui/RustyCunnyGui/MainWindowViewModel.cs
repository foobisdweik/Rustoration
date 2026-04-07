using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Platform.Storage;
using System.Collections.ObjectModel;

namespace RustyCunnyGui
{
    public partial class MainWindowViewModel : ObservableObject
    {
        [ObservableProperty]
        private string _inputPath = string.Empty;

        [ObservableProperty]
        private string _outputPath = string.Empty;

        [ObservableProperty]
        private string _prompt = "high quality restoration, detailed texture";

        [ObservableProperty]
        private string _logs = string.Empty;

        [ObservableProperty]
        private bool _isProcessing;

        private Process? _process;

        public MainWindowViewModel()
        {
            // Default paths for testing
            InputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "input");
            OutputPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");
        }

        [RelayCommand]
        private async Task BrowseInput(IStorageProvider storageProvider)
        {
            var result = await storageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "Select Input Directory",
                AllowMultiple = false
            });

            if (result.Count > 0)
            {
                InputPath = result[0].Path.LocalPath;
            }
        }

        [RelayCommand]
        private async Task BrowseOutput(IStorageProvider storageProvider)
        {
            var result = await storageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = "Select Output Directory",
                AllowMultiple = false
            });

            if (result.Count > 0)
            {
                OutputPath = result[0].Path.LocalPath;
            }
        }

        [RelayCommand(CanExecute = nameof(CanStart))]
        private async Task StartRestoration()
        {
            IsProcessing = true;
            Logs = "Starting restoration pipeline...\n";

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "restore.exe",
                    Arguments = $"-i \"{InputPath}\" -o \"{OutputPath}\" -p \"{Prompt}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
                
                _process.OutputDataReceived += (s, e) => AppendLog(e.Data);
                _process.ErrorDataReceived += (s, e) => AppendLog(e.Data, true);

                if (_process.Start())
                {
                    _process.BeginOutputReadLine();
                    _process.BeginErrorReadLine();
                    await _process.WaitForExitAsync();
                    AppendLog("Process exited with code: " + _process.ExitCode);
                }
                else
                {
                    AppendLog("Failed to start process.", true);
                }
            }
            catch (Exception ex)
            {
                AppendLog($"Exception: {ex.Message}", true);
            }
            finally
            {
                IsProcessing = false;
                _process = null;
            }
        }

        private bool CanStart() => !IsProcessing && !string.IsNullOrWhiteSpace(InputPath);

        private void AppendLog(string? message, bool isError = false)
        {
            if (string.IsNullOrEmpty(message)) return;
            
            var prefix = isError ? "[ERROR] " : "";
            Logs += $"{DateTime.Now:HH:mm:ss} {prefix}{message}\n";
        }

        [RelayCommand]
        private void StopRestoration()
        {
            if (_process != null && !_process.HasExited)
            {
                _process.Kill(true);
                AppendLog("Restoration stopped by user.");
            }
        }
    }
}
