//! Programming-language keyword tables used to classify vocabulary tokens.

pub const LANGS: &[(&str, &[&str])] = &[
    ("rust", &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
        "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
        "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while",
        "println", "vec", "Vec", "Option", "Some", "None", "Result", "Ok", "Err", "String", "usize", "isize",
        "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "str", "Box", "Rc", "Arc",
        "unwrap", "derive", "macro_rules",
    ]),
    ("python", &[
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
        "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield", "self",
        "print", "len", "range", "__init__", "__name__", "__main__", "isinstance", "kwargs", "args", "dict",
        "list", "tuple", "elif",
    ]),
    ("javascript", &[
        "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else",
        "export", "extends", "finally", "for", "function", "if", "import", "in", "instanceof", "let", "new",
        "return", "super", "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield",
        "async", "await", "null", "undefined", "true", "false", "console", "document", "window", "require",
        "module", "exports", "Promise", "prototype", "interface", "implements", "enum", "readonly", "namespace",
        "declare", "keyof", "boolean", "number", "string", "any", "unknown", "never", "useState", "useEffect",
    ]),
    ("c_cpp", &[
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum",
        "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
        "while", "class", "namespace", "template", "typename", "virtual", "public", "private", "protected",
        "nullptr", "std", "cout", "endl", "include", "#include", "#define", "#ifdef", "#endif", "#ifndef",
        "malloc", "free", "printf", "size_t", "uint8_t", "int32_t", "uint32_t", "int64_t", "constexpr",
        "operator", "friend", "mutable", "explicit", "noexcept",
    ]),
    ("java", &[
        "abstract", "boolean", "break", "byte", "case", "catch", "char", "class", "continue", "default", "do",
        "double", "else", "enum", "extends", "final", "finally", "float", "for", "if", "implements", "import",
        "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public",
        "return", "short", "static", "super", "switch", "synchronized", "this", "throw", "throws", "try",
        "void", "volatile", "while", "String", "System", "Override", "@Override", "ArrayList", "HashMap",
    ]),
    ("go", &[
        "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func",
        "go", "goto", "if", "import", "interface", "map", "package", "range", "return", "select", "struct",
        "switch", "type", "var", "nil", "err", "fmt", "Println", "Printf", "Errorf", "make", "append",
    ]),
    ("sql", &[
        "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE",
        "DROP", "ALTER", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "GROUP", "BY", "ORDER", "HAVING",
        "LIMIT", "OFFSET", "AND", "OR", "NOT", "NULL", "AS", "DISTINCT", "COUNT", "UNION", "PRIMARY", "KEY",
        "FOREIGN", "REFERENCES", "INDEX", "VARCHAR", "INTEGER", "DEFAULT", "select", "from", "where", "join",
    ]),
    ("shell", &[
        "echo", "fi", "then", "elif", "esac", "done", "do", "export", "sudo", "grep", "awk", "sed", "chmod",
        "chown", "mkdir", "cd", "ls", "rm", "cp", "mv", "cat", "curl", "wget", "apt", "pip", "npm", "git",
        "bash", "#!/bin/bash", "$HOME", "$PATH", "xargs", "unset", "source",
    ]),
];

/// Multi-character operators/symbols that strongly suggest code.
pub const CODE_SYMBOLS: &[&str] = &[
    "=>", "->", "::", "!=", "==", "===", "!==", "&&", "||", "<=", ">=", "+=", "-=", "*=", "/=", "++", "--",
    "<<", ">>", "/*", "*/", "//", "</", "/>", "#[", "](", ");", "){", "};", "()", "[]", "{}", "${", "?.", "??",
    ":=", "...", "<-",
];
