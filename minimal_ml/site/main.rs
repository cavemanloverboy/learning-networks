use std::f64::consts::PI;

use egui::{
    panel::Side,
    plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoints},
    Color32, FontFamily, FontId, RichText, ScrollArea,
};
use minimal_ml::site::LearningNetworks;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let options = eframe::WebOptions::default();

    eframe::start_web(
        "learning networks",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    )
    .expect("failed to start eframe");
}
