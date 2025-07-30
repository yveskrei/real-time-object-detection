// Custom modules
use client::config::{Config, Environment};
use client::inference::{InferenceModel, Precision};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Iniaitlize config
    //let app_config = Config::new(true, Environment::Production).expect("Error loading config");

    // Initiate model
    let inference = InferenceModel::new(
        "yolov9".to_string(), 
        "1".to_string(), 
        "images".to_string(), 
        "output".to_string(), 
        Precision::FP16
    ).await.unwrap();

    //Load instances
    inference.load_model(1).await.unwrap();
}
