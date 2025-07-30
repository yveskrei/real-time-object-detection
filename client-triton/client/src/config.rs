use dotenvy::from_path;
use std::path::Path;
use std::io::Error;
use std::env;
use std::process::Command;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Environment {
    Production,
    NonProduction
}

pub struct Config {
    local: bool,
    environment: Environment,
    gpu_name: String,
    source_ids: Vec<String>
}

impl Config {
    pub fn new(local: bool, environment: Environment) -> Result<Self, Error> {
        if local {
            // Load variables from local env file
            let base_dir = Path::new(file!()).parent().unwrap();
            let env_file = match environment {
                Environment::Production => ".env_test",
                Environment::NonProduction => ".env_test"
            };

            let env_path = base_dir.join(format!("../secrets/{}", env_file)).canonicalize().unwrap();
            from_path(env_path).expect("Error loading local env file");
        }

        // Extract variables from environment
        let source_ids: Vec<String> = env::var("SOURCE_IDS")
        .expect("SOURCES_IDS variable not found")
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

        Ok(Self {
            local,
            environment,
            source_ids,
            gpu_name: Config::get_gpu()
        })
    }

    fn get_gpu() -> String {
        let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .expect("failed to execute nvidia-smi");

        String::from_utf8_lossy(&output.stdout).to_string()
    }
}