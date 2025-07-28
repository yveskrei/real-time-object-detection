// Custom modules
use client::config::{Config, Environment};

fn main() {
    // Iniaitlize config
    let app_config = Config::new(true, Environment::Production).expect("Error loading config");

}
