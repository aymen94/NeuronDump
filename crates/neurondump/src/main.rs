//! The NeuronDump binary: wires the core library to the CLI, web and trace crates.

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("NEURONDUMP_LOG"))
        .with_writer(std::io::stderr)
        .init();

    #[cfg(feature = "trace")]
    neurondump_core::trace::register_backend(Box::new(neurondump_trace::CandleBackend::new()));

    #[cfg(feature = "web")]
    let web: Option<neurondump_cli::WebLauncher> = Some(std::sync::Arc::new(|app, args: neurondump_cli::WebArgs| {
        neurondump_web::serve(app, neurondump_web::ServeOptions { host: args.host, port: args.port, open: args.open })
    }));
    #[cfg(not(feature = "web"))]
    let web: Option<neurondump_cli::WebLauncher> = None;

    // Windows gives the main thread 1 MiB of stack, which clap's large derive and the
    // REPL overflow in debug builds; run everything on a thread with a bigger stack.
    let code = std::thread::Builder::new()
        .name("neurondump".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || neurondump_cli::main(web))
        .expect("spawn main thread")
        .join()
        .unwrap_or(101);
    std::process::exit(code);
}
