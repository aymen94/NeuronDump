//! The embedded frontend (`frontend/`) served with the right content types.

use super::*;

#[derive(RustEmbed)]
#[folder = "frontend/"]
pub(super) struct Assets;

pub(super) async fn static_file(Path(path): Path<String>) -> Response {
    serve_asset(&path)
}

pub(super) async fn index() -> Response {
    serve_asset("index.html")
}

pub(super) fn serve_asset(path: &str) -> Response {
    if path.contains("..") {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match Assets::get(path) {
        Some(f) => {
            let mime = match path.rsplit('.').next() {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "text/javascript; charset=utf-8",
                Some("css") => "text/css; charset=utf-8",
                Some("svg") => "image/svg+xml",
                Some("wasm") => "application/wasm",
                Some("json") => "application/json",
                _ => "application/octet-stream",
            };
            ([(header::CONTENT_TYPE, mime)], f.data.into_owned()).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
