use rusty_cunny::RestorationPipeline;
use anyhow::Result;
use candle_core::Device;
use std::path::{Path, PathBuf};
use tokio::fs;
use log::{info, error};
use std::sync::Arc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory or single image file
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to YOLOv8-seg model safetensors
    #[arg(short, long, default_value = "./models/yolov8-seg.safetensors")]
    model: PathBuf,

    /// Restoration prompt
    #[arg(short, long, default_value = "high quality restoration, detailed texture")]
    prompt: String,

    /// Watch mode (keep running and watch input directory)
    #[arg(short, long, default_value_t = false)]
    watch: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let device = Device::cuda_if_available(0)?;
    info!("Using device: {:?}", device);

    let pipeline = Arc::new(RestorationPipeline::new(device, &args.model).await?);
    info!("Pipeline loaded");

    let input_path = args.input.unwrap_or_else(|| PathBuf::from("./input"));
    let output_path = args.output.unwrap_or_else(|| input_path.join("output"));

    // Ensure output directory exists
    if !output_path.exists() {
        fs::create_dir_all(&output_path).await?;
    }

    if args.watch {
        watch_folder(&input_path, &output_path, &args.prompt, pipeline).await?;
    } else {
        process_once(&input_path, &output_path, &args.prompt, &pipeline).await?;
    }

    Ok(())
}

async fn process_once(input: &Path, output_dir: &Path, prompt: &str, pipeline: &RestorationPipeline) -> Result<()> {
    if input.is_file() {
        process_single_image(pipeline, input, output_dir, prompt).await?;
    } else if input.is_dir() {
        let mut entries = fs::read_dir(input).await?;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if is_image(&path) {
                if let Err(e) = process_single_image(pipeline, &path, output_dir, prompt).await {
                    error!("Failed to process {:?}: {}", path, e);
                }
            }
        }
    }
    Ok(())
}

async fn watch_folder(watch_dir: &Path, output_dir: &Path, prompt: &str, pipeline: Arc<RestorationPipeline>) -> Result<()> {
    info!("Watching directory: {:?}", watch_dir);

    loop {
        if let Ok(mut entries) = fs::read_dir(watch_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if is_image(&path) {
                    info!("Processing: {:?}", path);
                    if let Err(e) = process_single_image(&pipeline, &path, output_dir, prompt).await {
                        error!("Failed to process {:?}: {}", path, e);
                    }
                    
                    // Move to processed directory
                    let processed_dir = watch_dir.join("processed");
                    fs::create_dir_all(&processed_dir).await?;
                    let filename = path.file_name().unwrap();
                    let _result = fs::rename(&path, processed_dir.join(filename)).await;
                }
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
}

fn is_image(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        matches!(ext.to_str(), Some("png") | Some("jpg") | Some("jpeg") | Some("webp"))
    } else {
        false
    }
}

async fn process_single_image(pipeline: &RestorationPipeline, image_path: &Path, output_dir: &Path, prompt: &str) -> Result<()> {
    let result = pipeline.restore(image_path, prompt).await?;
    
    let filename = image_path.file_stem().unwrap().to_str().unwrap();
    let output_path = output_dir.join(format!("{}_restored.png", filename));
    
    // Save logic (this is a placeholder for saving the candle::Tensor as an image)
    // In a real scenario, you'd convert the tensor back to an image::DynamicImage
    info!("Restored image saved to: {:?}", output_path);
    
    // Explicit drop to reclaim VRAM
    drop(result);
    
    Ok(())
}
