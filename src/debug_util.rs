// 通用调试输出宏，设置环境变量 DEBUG=1 时生效
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").map(|v| v == "1").unwrap_or(false) {
            eprintln!($($arg)*);
        }
    };
} 