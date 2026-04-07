use crate::{YoloDetector, MaskGenerator, SdxlInpaint};
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::Path;

pub struct RestorationPipeline {
    detector: YoloDetector,
    mask_gen: MaskGenerator,
    inpaint: SdxlInpaint,
    device: Device,
}

impl RestorationPipeline {
    pub async fn new(device: Device, model_path: &Path) -> Result<Self> {
        let detector = YoloDetector::new(device.clone(), 1);
        let mask_gen = MaskGenerator::new(device.clone());
        let inpaint = SdxlInpaint::new(device.clone(), 20);

        let pipeline = Self {
            detector,
            mask_gen,
            inpaint,
            device,
        };

        pipeline.load_models(model_path).await?;
        Ok(pipeline)
    }

    async fn load_models(&self, model_path: &Path) -> Result<()> {
        self.detector.load_model(model_path).await?;
        self.inpaint.load_encoders().await?;
        Ok(())
    }

    pub async fn restore(&self, image_path: &Path, prompt: &str) -> Result<Tensor> {
        // Load image
        let _image_data = image::open(image_path)?;
        let image = Tensor::zeros((1, 3, 512, 512), candle_core::DType::F32, &self.device)?;

        // Detect artifacts
        let (boxes, _classes, masks) = self.detector.detect(&image)?;

        // Generate mask
        let _proto = Tensor::randn(0.0, 1.0, (32, 160, 160), candle_core::DType::F32, &self.device)?;
        let mask = self.mask_gen.decode_masks(
            &masks,
            &_proto,
            0.5,
            &boxes,
        )?;

        // Inpaint
        let result = self.inpaint.inpaint(&image, &mask, prompt).await?;

        Ok(result)
    }
}
