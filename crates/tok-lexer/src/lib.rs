/// Tok language lexer
///
/// Tokenizes Tok source code into a flat token stream.
/// Key design decisions:
/// - String interpolation `"hello {expr}"` produces StringStart/StringMid/StringEnd tokens
/// - `#` at line start or after operators = comment; otherwise = length operator
/// - After `.` token, suppress greedy float parsing (for `t.1.1` chained tuple access)
/// - Type names (i, f, s, b, a) are lexed as Ident — disambiguation is the parser's job

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),    // plain string, no interpolation
    RawStr(String), // backtick string
    True,
    False,
    Nil,

    // String interpolation segments
    StringStart(String), // text before first {
    StringMid(String),   // text between } and next {
    StringEnd(String),   // text after last }

    // Identifiers
    Ident(String),

    // Keywords
    Func, // f (as keyword, at statement start before ident)
    Go,   // go
    Sel,  // sel

    // Operators — arithmetic
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    Percent,  // %
    StarStar, // **

    // Operators — comparison
    EqEq,   // ==
    BangEq, // !=
    Gt,     // >
    Lt,     // <
    GtEq,   // >=
    LtEq,   // <=

    // Operators — logical
    Amp,  // & (logical and)
    Pipe, // | (logical or)
    Bang, // ! (not / break)

    // Operators — bitwise
    AmpAmp,     // && (bitwise and)
    PipePipe,   // || (bitwise or)
    CaretCaret, // ^^ (bitwise xor)
    LtLt,       // << (append)
    LtLtEq,     // <<= (append-assign)
    GtGt,       // >> (shift right)

    // Operators — control/special
    Question,         // ?
    QuestionEq,       // ?=
    QuestionCaret,    // ?^
    QuestionQuestion, // ??
    QuestionGt,       // ?>
    Tilde,            // ~
    Caret,            // ^ (early return)
    DotQuestion,      // .?
    DotDot,           // ..
    DotDotEq,         // ..=

    // Operators — pipe/reduce
    PipeGt,  // |>
    SlashGt, // />

    // Operators — assignment
    Eq,         // =
    PlusEq,     // +=
    MinusEq,    // -=
    StarEq,     // *=
    SlashEq,    // /=
    PercentEq,  // %=
    StarStarEq, // **=

    // Operators — channel
    ArrowLeft,  // <- (receive/send)
    ArrowRight, // -> (unused currently but reserved)

    // Operators — other
    Hash,   // # (length prefix)
    At,     // @
    GtBang, // >! (continue)

    // Structural
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
    Semi,      // ;
    Colon,     // :
    Dot,       // .
    Backslash, // \

    // Separators
    Newline, // significant newline

    // End
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Token::Int(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::Str(s) => write!(f, "\"{}\"", s),
            Token::RawStr(s) => write!(f, "`{}`", s),
            Token::True => write!(f, "T"),
            Token::False => write!(f, "F"),
            Token::Nil => write!(f, "N"),
            Token::StringStart(s) => write!(f, "\"{}{{", s),
            Token::StringMid(s) => write!(f, "}}{}{{", s),
            Token::StringEnd(s) => write!(f, "}}{}\"", s),
            Token::Ident(s) => write!(f, "{}", s),
            Token::Func => write!(f, "f"),
            Token::Go => write!(f, "go"),
            Token::Sel => write!(f, "sel"),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.msg)
    }
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize()
}

struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    tokens: Vec<Token>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Lexer {
            source: source.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.source.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.source.get(self.pos).copied();
        if let Some(c) = ch {
            self.pos += 1;
            if c == b'\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        ch
    }

    fn err(&self, msg: impl Into<String>) -> LexError {
        LexError {
            msg: msg.into(),
            line: self.line,
            col: self.col,
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn is_at_line_start(&self) -> bool {
        // Check if we're at the start of a line (only whitespace before current position on this line)
        if self.pos == 0 {
            return true;
        }
        let mut i = self.pos - 1;
        loop {
            let ch = self.source[i];
            if ch == b'\n' {
                return true;
            }
            if ch != b' ' && ch != b'\t' && ch != b'\r' {
                return false;
            }
            if i == 0 {
                return true;
            }
            i -= 1;
        }
    }

    fn last_meaningful_token(&self) -> Option<&Token> {
        self.tokens
            .iter()
            .rev()
            .find(|t| !matches!(t, Token::Newline))
    }

    fn hash_is_comment(&self) -> bool {
        // # is a comment if:
        // 1. At the start of a line
        // 2. After a newline token (with possible whitespace)
        // 3. After no tokens at all
        // # is the length operator if it appears in expression context
        if self.is_at_line_start() {
            return true;
        }
        match self.last_meaningful_token() {
            None => true,
            Some(tok) => matches!(tok, Token::Semi | Token::LBrace | Token::Newline),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        loop {
            self.skip_whitespace();

            let ch = match self.peek() {
                None => break,
                Some(c) => c,
            };

            match ch {
                b'\n' => {
                    self.advance();
                    // Collapse multiple newlines; don't add if last token is already Newline or start
                    if !self.tokens.is_empty() {
                        if let Some(last) = self.tokens.last() {
                            if !matches!(last, Token::Newline | Token::LBrace | Token::Semi) {
                                self.tokens.push(Token::Newline);
                            }
                        }
                    }
                }

                b'#' => {
                    if self.hash_is_comment() {
                        // Comment — skip to end of line
                        while let Some(c) = self.peek() {
                            if c == b'\n' {
                                break;
                            }
                            self.advance();
                        }
                    } else {
                        // Length operator
                        self.advance();
                        self.tokens.push(Token::Hash);
                    }
                }

                b';' => {
                    self.advance();
                    // Treat like newline — collapse
                    if !self.tokens.is_empty() {
                        if let Some(last) = self.tokens.last() {
                            if !matches!(last, Token::Newline | Token::Semi | Token::LBrace) {
                                self.tokens.push(Token::Newline);
                            }
                        }
                    }
                }

                b'"' => {
                    self.lex_string()?;
                }

                b'`' => {
                    self.lex_raw_string()?;
                }

                b'(' => {
                    self.advance();
                    self.tokens.push(Token::LParen);
                }
                b')' => {
                    self.advance();
                    self.tokens.push(Token::RParen);
                }
                b'[' => {
                    self.advance();
                    self.tokens.push(Token::LBracket);
                }
                b']' => {
                    self.advance();
                    self.tokens.push(Token::RBracket);
                }
                b'{' => {
                    self.advance();
                    self.tokens.push(Token::LBrace);
                }
                b'}' => {
                    self.advance();
                    self.tokens.push(Token::RBrace);
                }
                b':' => {
                    self.advance();
                    self.tokens.push(Token::Colon);
                }
                b'@' => {
                    self.advance();
                    self.tokens.push(Token::At);
                }
                b'\\' => {
                    self.advance();
                    self.tokens.push(Token::Backslash);
                }
                b'~' => {
                    self.advance();
                    self.tokens.push(Token::Tilde);
                }

                b'+' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::PlusEq);
                    } else {
                        self.tokens.push(Token::Plus);
                    }
                }

                b'-' => {
                    self.advance();
                    if self.peek() == Some(b'>') {
                        self.advance();
                        self.tokens.push(Token::ArrowRight);
                    } else if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::MinusEq);
                    } else {
                        self.tokens.push(Token::Minus);
                    }
                }

                b'*' => {
                    self.advance();
                    if self.peek() == Some(b'*') {
                        self.advance();
                        if self.peek() == Some(b'=') {
                            self.advance();
                            self.tokens.push(Token::StarStarEq);
                        } else {
                            self.tokens.push(Token::StarStar);
                        }
                    } else if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::StarEq);
                    } else {
                        self.tokens.push(Token::Star);
                    }
                }

                b'/' => {
                    self.advance();
                    if self.peek() == Some(b'/') {
                        // // comment — skip to end of line
                        while let Some(c) = self.peek() {
                            if c == b'\n' {
                                break;
                            }
                            self.advance();
                        }
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        self.tokens.push(Token::SlashGt);
                    } else if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::SlashEq);
                    } else {
                        self.tokens.push(Token::Slash);
                    }
                }

                b'%' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::PercentEq);
                    } else {
                        self.tokens.push(Token::Percent);
                    }
                }

                b'=' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::EqEq);
                    } else {
                        self.tokens.push(Token::Eq);
                    }
                }

                b'!' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::BangEq);
                    } else {
                        self.tokens.push(Token::Bang);
                    }
                }

                b'>' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::GtEq);
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        self.tokens.push(Token::GtGt);
                    } else if self.peek() == Some(b'!') {
                        self.advance();
                        self.tokens.push(Token::GtBang);
                    } else {
                        self.tokens.push(Token::Gt);
                    }
                }

                b'<' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::LtEq);
                    } else if self.peek() == Some(b'<') {
                        self.advance();
                        if self.peek() == Some(b'=') {
                            self.advance();
                            self.tokens.push(Token::LtLtEq);
                        } else {
                            self.tokens.push(Token::LtLt);
                        }
                    } else if self.peek() == Some(b'-') {
                        self.advance();
                        self.tokens.push(Token::ArrowLeft);
                    } else {
                        self.tokens.push(Token::Lt);
                    }
                }

                b'&' => {
                    self.advance();
                    if self.peek() == Some(b'&') {
                        self.advance();
                        self.tokens.push(Token::AmpAmp);
                    } else {
                        self.tokens.push(Token::Amp);
                    }
                }

                b'|' => {
                    self.advance();
                    if self.peek() == Some(b'|') {
                        self.advance();
                        self.tokens.push(Token::PipePipe);
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        self.tokens.push(Token::PipeGt);
                    } else {
                        self.tokens.push(Token::Pipe);
                    }
                }

                b'^' => {
                    self.advance();
                    if self.peek() == Some(b'^') {
                        self.advance();
                        self.tokens.push(Token::CaretCaret);
                    } else {
                        self.tokens.push(Token::Caret);
                    }
                }

                b'?' => {
                    self.advance();
                    if self.peek() == Some(b'=') {
                        self.advance();
                        self.tokens.push(Token::QuestionEq);
                    } else if self.peek() == Some(b'^') {
                        self.advance();
                        self.tokens.push(Token::QuestionCaret);
                    } else if self.peek() == Some(b'?') {
                        self.advance();
                        self.tokens.push(Token::QuestionQuestion);
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        self.tokens.push(Token::QuestionGt);
                    } else {
                        self.tokens.push(Token::Question);
                    }
                }

                b'.' => {
                    self.advance();
                    if self.peek() == Some(b'.') {
                        self.advance();
                        if self.peek() == Some(b'=') {
                            self.advance();
                            self.tokens.push(Token::DotDotEq);
                        } else {
                            self.tokens.push(Token::DotDot);
                        }
                    } else if self.peek() == Some(b'?') {
                        self.advance();
                        self.tokens.push(Token::DotQuestion);
                    } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        // Leading-dot float like .5
                        // But NOT after an expression (then it's member access + int)
                        let is_after_expr = self.last_meaningful_token().is_some_and(|t| {
                            matches!(
                                t,
                                Token::Ident(_)
                                    | Token::Int(_)
                                    | Token::Float(_)
                                    | Token::Str(_)
                                    | Token::RawStr(_)
                                    | Token::True
                                    | Token::False
                                    | Token::Nil
                                    | Token::RParen
                                    | Token::RBracket
                                    | Token::RBrace
                                    | Token::StringEnd(_)
                            )
                        });
                        if is_after_expr {
                            // Member access followed by integer (tuple index)
                            self.tokens.push(Token::Dot);
                            // The digit(s) will be lexed as Int in the next iteration
                        } else {
                            // Leading-dot float: .5, .123
                            let num = self.lex_float_after_dot()?;
                            self.tokens.push(Token::Float(num));
                        }
                    } else {
                        self.tokens.push(Token::Dot);
                    }
                }

                c if c.is_ascii_digit() => {
                    self.lex_number()?;
                }

                c if c.is_ascii_alphabetic() || c == b'_' => {
                    self.lex_identifier();
                }

                c if c > 127 => {
                    // Non-ASCII UTF-8 byte — skip the full codepoint
                    // UTF-8: 110xxxxx = 2 bytes, 1110xxxx = 3 bytes, 11110xxx = 4 bytes
                    let extra = if c >= 0xF0 {
                        3
                    } else if c >= 0xE0 {
                        2
                    } else {
                        1
                    };
                    self.advance();
                    for _ in 0..extra {
                        if self.peek().is_some_and(|b| (0x80..0xC0).contains(&b)) {
                            self.advance();
                        }
                    }
                    return Err(self.err("unexpected non-ASCII character".to_string()));
                }

                _ => {
                    return Err(self.err(format!("unexpected character: '{}'", ch as char)));
                }
            }
        }

        // Remove trailing newlines
        while self.tokens.last() == Some(&Token::Newline) {
            self.tokens.pop();
        }

        self.tokens.push(Token::Eof);
        Ok(self.tokens.clone())
    }

    /// Collect source text from `start` to current position, filtering out underscores.
    fn collect_text(&self, start: usize) -> String {
        self.source[start..self.pos]
            .iter()
            .filter(|&&c| c != b'_')
            .map(|&c| c as char)
            .collect()
    }

    /// Consume an optional exponent suffix (e/E followed by optional +/- and digits).
    fn consume_exponent(&mut self) {
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.advance();
            }
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }
    }

    /// Parse a radix integer literal (hex, binary, or octal).
    /// `valid` tests if a byte is a valid digit for this radix.
    fn lex_radix_int(
        &mut self,
        valid: fn(u8) -> bool,
        radix: u32,
        prefix: &str,
    ) -> Result<(), LexError> {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if valid(ch) || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        let text = self.collect_text(start);
        if text.is_empty() {
            return Err(self.err(format!("expected digits after {}", prefix)));
        }
        let val = i64::from_str_radix(&text, radix)
            .map_err(|_| self.err(format!("invalid {} literal: {}{}", prefix, prefix, text)))?;
        self.tokens.push(Token::Int(val));
        Ok(())
    }

    fn lex_number(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        let first = self.advance().unwrap();

        // Check for hex/binary/octal prefixes
        if first == b'0' {
            match self.peek() {
                Some(b'x') | Some(b'X') => {
                    self.advance();
                    return self.lex_radix_int(|c| c.is_ascii_hexdigit(), 16, "0x");
                }
                Some(b'b') | Some(b'B') => {
                    self.advance();
                    return self.lex_radix_int(|c| c == b'0' || c == b'1', 2, "0b");
                }
                Some(b'o') | Some(b'O') => {
                    self.advance();
                    return self.lex_radix_int(|c| (b'0'..=b'7').contains(&c), 8, "0o");
                }
                _ => {}
            }
        }

        // Consume digits and underscores
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        // Check for float: decimal point
        // GOTCHA: suppress float parsing when previous token was Dot (for `t.1.1`)
        let prev_was_dot = self.tokens.last().is_some_and(|t| matches!(t, Token::Dot));

        if !prev_was_dot
            && self.peek() == Some(b'.')
            && self.peek_at(1).is_some_and(|c| c.is_ascii_digit())
        {
            // It's a float with decimal point
            self.advance(); // consume .
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() || ch == b'_' {
                    self.advance();
                } else {
                    break;
                }
            }
            self.consume_exponent();
            let text = self.collect_text(start);
            let val: f64 = text
                .parse()
                .map_err(|_| self.err(format!("invalid float: {}", text)))?;
            self.tokens.push(Token::Float(val));
        } else if !prev_was_dot && matches!(self.peek(), Some(b'e') | Some(b'E')) {
            // Scientific notation without decimal: 1e10
            self.consume_exponent();
            let text = self.collect_text(start);
            let val: f64 = text
                .parse()
                .map_err(|_| self.err(format!("invalid float: {}", text)))?;
            self.tokens.push(Token::Float(val));
        } else {
            // Integer
            let text = self.collect_text(start);
            let val: i64 = text
                .parse()
                .map_err(|_| self.err(format!("invalid integer: {}", text)))?;
            self.tokens.push(Token::Int(val));
        }

        Ok(())
    }

    fn lex_float_after_dot(&mut self) -> Result<f64, LexError> {
        // We've already consumed the '.', now consume digits
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        self.consume_exponent();
        let text = self.collect_text(start);
        let full = format!("0.{}", text);
        full.parse()
            .map_err(|_| self.err(format!("invalid float: .{}", text)))
    }

    fn lex_identifier(&mut self) {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos]
            .iter()
            .map(|&c| c as char)
            .collect();

        let token = match text.as_str() {
            "T" => Token::True,
            "F" => Token::False,
            "N" => Token::Nil,
            "go" => Token::Go,
            "sel" => Token::Sel,
            // `f` is only a keyword when followed by an identifier (function decl)
            // We lex it as Func if the next non-whitespace char starts an identifier
            "f" => {
                // Look ahead past whitespace
                let mut look = self.pos;
                while look < self.source.len()
                    && (self.source[look] == b' ' || self.source[look] == b'\t')
                {
                    look += 1;
                }
                if look < self.source.len()
                    && (self.source[look].is_ascii_alphabetic() || self.source[look] == b'_')
                {
                    Token::Func
                } else {
                    Token::Ident(text)
                }
            }
            _ => Token::Ident(text),
        };

        self.tokens.push(token);
    }

    fn lex_string(&mut self) -> Result<(), LexError> {
        self.advance(); // consume opening "

        let mut buf = String::new();
        let mut has_interpolation = false;

        loop {
            match self.peek() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.advance();
                    if has_interpolation {
                        self.tokens.push(Token::StringEnd(buf));
                    } else {
                        self.tokens.push(Token::Str(buf));
                    }
                    return Ok(());
                }
                Some(b'\\') => {
                    self.advance();
                    let escaped = self.lex_escape()?;
                    buf.push(escaped);
                }
                Some(b'{') => {
                    self.advance();
                    // String interpolation
                    if !has_interpolation {
                        self.tokens.push(Token::StringStart(buf));
                        has_interpolation = true;
                    } else {
                        self.tokens.push(Token::StringMid(buf));
                    }
                    buf = String::new();
                    // Lex tokens until matching }
                    self.lex_interpolation_expr()?;
                }
                Some(b'\n') => {
                    self.advance();
                    buf.push('\n');
                }
                Some(c) => {
                    self.advance();
                    buf.push(c as char);
                }
            }
        }
    }

    fn lex_escape(&mut self) -> Result<char, LexError> {
        match self.advance() {
            None => Err(self.err("unterminated escape sequence")),
            Some(b'n') => Ok('\n'),
            Some(b't') => Ok('\t'),
            Some(b'\\') => Ok('\\'),
            Some(b'"') => Ok('"'),
            Some(b'{') => Ok('{'),
            Some(b'0') => Ok('\0'),
            Some(b'x') => {
                let h1 = self
                    .advance()
                    .ok_or_else(|| self.err("incomplete \\x escape"))?;
                let h2 = self
                    .advance()
                    .ok_or_else(|| self.err("incomplete \\x escape"))?;
                let hex = format!("{}{}", h1 as char, h2 as char);
                let code = u8::from_str_radix(&hex, 16)
                    .map_err(|_| self.err(format!("invalid hex escape: \\x{}", hex)))?;
                Ok(code as char)
            }
            Some(b'u') => {
                if self.advance() != Some(b'{') {
                    return Err(self.err("expected '{' after \\u"));
                }
                let mut hex = String::new();
                loop {
                    match self.peek() {
                        Some(b'}') => {
                            self.advance();
                            break;
                        }
                        Some(c) if (c as char).is_ascii_hexdigit() => {
                            self.advance();
                            hex.push(c as char);
                        }
                        _ => return Err(self.err("invalid unicode escape")),
                    }
                }
                let code = u32::from_str_radix(&hex, 16)
                    .map_err(|_| self.err(format!("invalid unicode escape: \\u{{{}}}", hex)))?;
                char::from_u32(code)
                    .ok_or_else(|| self.err(format!("invalid unicode codepoint: {}", code)))
            }
            Some(c) => Err(self.err(format!("unknown escape: \\{}", c as char))),
        }
    }

    fn lex_interpolation_expr(&mut self) -> Result<(), LexError> {
        // Lex tokens until we find a matching } at depth 0
        let mut depth = 0u32;
        loop {
            self.skip_whitespace();
            match self.peek() {
                None => return Err(self.err("unterminated string interpolation")),
                Some(b'}') => {
                    if depth == 0 {
                        self.advance(); // consume closing }
                        return Ok(());
                    }
                    self.advance();
                    self.tokens.push(Token::RBrace);
                    depth -= 1;
                }
                Some(b'{') => {
                    self.advance();
                    self.tokens.push(Token::LBrace);
                    depth += 1;
                }
                Some(b'"') => {
                    // Nested string inside interpolation
                    self.lex_string()?;
                }
                Some(b'`') => {
                    self.lex_raw_string()?;
                }
                Some(b'\n') => {
                    self.advance();
                    // Treat as separator inside interpolation
                }
                _ => {
                    // Re-use main lexer logic for operators and other tokens
                    let ch = self.peek().unwrap();
                    match ch {
                        b'(' => {
                            self.advance();
                            self.tokens.push(Token::LParen);
                        }
                        b')' => {
                            self.advance();
                            self.tokens.push(Token::RParen);
                        }
                        b'[' => {
                            self.advance();
                            self.tokens.push(Token::LBracket);
                        }
                        b']' => {
                            self.advance();
                            self.tokens.push(Token::RBracket);
                        }
                        b':' => {
                            self.advance();
                            self.tokens.push(Token::Colon);
                        }
                        b'@' => {
                            self.advance();
                            self.tokens.push(Token::At);
                        }
                        b'+' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::PlusEq);
                            } else {
                                self.tokens.push(Token::Plus);
                            }
                        }
                        b'-' => {
                            self.advance();
                            if self.peek() == Some(b'>') {
                                self.advance();
                                self.tokens.push(Token::ArrowRight);
                            } else if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::MinusEq);
                            } else {
                                self.tokens.push(Token::Minus);
                            }
                        }
                        b'*' => {
                            self.advance();
                            if self.peek() == Some(b'*') {
                                self.advance();
                                if self.peek() == Some(b'=') {
                                    self.advance();
                                    self.tokens.push(Token::StarStarEq);
                                } else {
                                    self.tokens.push(Token::StarStar);
                                }
                            } else if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::StarEq);
                            } else {
                                self.tokens.push(Token::Star);
                            }
                        }
                        b'/' => {
                            self.advance();
                            if self.peek() == Some(b'/') {
                                // // comment — skip to end of line
                                while let Some(c) = self.peek() {
                                    if c == b'\n' {
                                        break;
                                    }
                                    self.advance();
                                }
                            } else if self.peek() == Some(b'>') {
                                self.advance();
                                self.tokens.push(Token::SlashGt);
                            } else if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::SlashEq);
                            } else {
                                self.tokens.push(Token::Slash);
                            }
                        }
                        b'%' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::PercentEq);
                            } else {
                                self.tokens.push(Token::Percent);
                            }
                        }
                        b'=' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::EqEq);
                            } else {
                                self.tokens.push(Token::Eq);
                            }
                        }
                        b'!' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::BangEq);
                            } else {
                                self.tokens.push(Token::Bang);
                            }
                        }
                        b'>' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::GtEq);
                            } else if self.peek() == Some(b'>') {
                                self.advance();
                                self.tokens.push(Token::GtGt);
                            } else if self.peek() == Some(b'!') {
                                self.advance();
                                self.tokens.push(Token::GtBang);
                            } else {
                                self.tokens.push(Token::Gt);
                            }
                        }
                        b'<' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::LtEq);
                            } else if self.peek() == Some(b'<') {
                                self.advance();
                                if self.peek() == Some(b'=') {
                                    self.advance();
                                    self.tokens.push(Token::LtLtEq);
                                } else {
                                    self.tokens.push(Token::LtLt);
                                }
                            } else if self.peek() == Some(b'-') {
                                self.advance();
                                self.tokens.push(Token::ArrowLeft);
                            } else {
                                self.tokens.push(Token::Lt);
                            }
                        }
                        b'&' => {
                            self.advance();
                            if self.peek() == Some(b'&') {
                                self.advance();
                                self.tokens.push(Token::AmpAmp);
                            } else {
                                self.tokens.push(Token::Amp);
                            }
                        }
                        b'|' => {
                            self.advance();
                            if self.peek() == Some(b'|') {
                                self.advance();
                                self.tokens.push(Token::PipePipe);
                            } else if self.peek() == Some(b'>') {
                                self.advance();
                                self.tokens.push(Token::PipeGt);
                            } else {
                                self.tokens.push(Token::Pipe);
                            }
                        }
                        b'^' => {
                            self.advance();
                            if self.peek() == Some(b'^') {
                                self.advance();
                                self.tokens.push(Token::CaretCaret);
                            } else {
                                self.tokens.push(Token::Caret);
                            }
                        }
                        b'?' => {
                            self.advance();
                            if self.peek() == Some(b'=') {
                                self.advance();
                                self.tokens.push(Token::QuestionEq);
                            } else if self.peek() == Some(b'^') {
                                self.advance();
                                self.tokens.push(Token::QuestionCaret);
                            } else if self.peek() == Some(b'?') {
                                self.advance();
                                self.tokens.push(Token::QuestionQuestion);
                            } else if self.peek() == Some(b'>') {
                                self.advance();
                                self.tokens.push(Token::QuestionGt);
                            } else {
                                self.tokens.push(Token::Question);
                            }
                        }
                        b'.' => {
                            self.advance();
                            if self.peek() == Some(b'.') {
                                self.advance();
                                if self.peek() == Some(b'=') {
                                    self.advance();
                                    self.tokens.push(Token::DotDotEq);
                                } else {
                                    self.tokens.push(Token::DotDot);
                                }
                            } else if self.peek() == Some(b'?') {
                                self.advance();
                                self.tokens.push(Token::DotQuestion);
                            } else {
                                self.tokens.push(Token::Dot);
                            }
                        }
                        b'#' => {
                            self.advance();
                            self.tokens.push(Token::Hash);
                        }
                        c if c.is_ascii_digit() => {
                            self.lex_number()?;
                        }
                        c if c.is_ascii_alphabetic() || c == b'_' => {
                            self.lex_identifier();
                        }
                        _ => {
                            return Err(self.err(format!(
                                "unexpected character in interpolation: '{}'",
                                ch as char
                            )));
                        }
                    }
                }
            }
        }
    }

    fn lex_raw_string(&mut self) -> Result<(), LexError> {
        self.advance(); // consume opening `
        let mut buf = String::new();
        loop {
            match self.advance() {
                None => return Err(self.err("unterminated raw string")),
                Some(b'`') => {
                    self.tokens.push(Token::RawStr(buf));
                    return Ok(());
                }
                Some(c) => buf.push(c as char),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let tokens = lex("x=5").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Eq,
                Token::Int(5),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_float() {
        let tokens = lex("3.14").unwrap();
        assert_eq!(tokens, vec![Token::Float(3.14), Token::Eof]);
    }

    #[test]
    fn test_leading_dot_float() {
        let tokens = lex(".5").unwrap();
        assert_eq!(tokens, vec![Token::Float(0.5), Token::Eof]);
    }

    #[test]
    fn test_hex() {
        let tokens = lex("0xff").unwrap();
        assert_eq!(tokens, vec![Token::Int(255), Token::Eof]);
    }

    #[test]
    fn test_binary() {
        let tokens = lex("0b1010").unwrap();
        assert_eq!(tokens, vec![Token::Int(10), Token::Eof]);
    }

    #[test]
    fn test_octal() {
        let tokens = lex("0o77").unwrap();
        assert_eq!(tokens, vec![Token::Int(63), Token::Eof]);
    }

    #[test]
    fn test_underscore_separators() {
        let tokens = lex("1_000_000").unwrap();
        assert_eq!(tokens, vec![Token::Int(1000000), Token::Eof]);
    }

    #[test]
    fn test_string() {
        let tokens = lex(r#""hello""#).unwrap();
        assert_eq!(tokens, vec![Token::Str("hello".into()), Token::Eof]);
    }

    #[test]
    fn test_string_interpolation() {
        let tokens = lex(r#""hello {name}""#).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::StringStart("hello ".into()),
                Token::Ident("name".into()),
                Token::StringEnd("".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_string_interpolation_expr() {
        let tokens = lex(r#""2+2={2+2}""#).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::StringStart("2+2=".into()),
                Token::Int(2),
                Token::Plus,
                Token::Int(2),
                Token::StringEnd("".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_raw_string() {
        let tokens = lex(r#"`raw {not interpolated}`"#).unwrap();
        assert_eq!(
            tokens,
            vec![Token::RawStr("raw {not interpolated}".into()), Token::Eof]
        );
    }

    #[test]
    fn test_keywords() {
        let tokens = lex("T F N").unwrap();
        assert_eq!(
            tokens,
            vec![Token::True, Token::False, Token::Nil, Token::Eof,]
        );
    }

    #[test]
    fn test_func_keyword() {
        let tokens = lex("f add(a b)=a+b").unwrap();
        assert_eq!(tokens[0], Token::Func);
    }

    #[test]
    fn test_func_as_ident() {
        // f used in type context (not followed by ident) should be Ident
        let tokens = lex("x:f=3.14").unwrap();
        assert_eq!(tokens[0], Token::Ident("x".into()));
        assert_eq!(tokens[2], Token::Ident("f".into()));
    }

    #[test]
    fn test_operators() {
        let tokens = lex("+ - * / % **").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::StarStar,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_comparison() {
        let tokens = lex("== != > < >= <=").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::EqEq,
                Token::BangEq,
                Token::Gt,
                Token::Lt,
                Token::GtEq,
                Token::LtEq,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_bitwise() {
        let tokens = lex("&& || ^^ << >>").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::AmpAmp,
                Token::PipePipe,
                Token::CaretCaret,
                Token::LtLt,
                Token::GtGt,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_pipe_filter_reduce() {
        let tokens = lex("|> ?> />").unwrap();
        assert_eq!(
            tokens,
            vec![Token::PipeGt, Token::QuestionGt, Token::SlashGt, Token::Eof,]
        );
    }

    #[test]
    fn test_channel_ops() {
        let tokens = lex("<- ->").unwrap();
        assert_eq!(
            tokens,
            vec![Token::ArrowLeft, Token::ArrowRight, Token::Eof]
        );
    }

    #[test]
    fn test_question_variants() {
        let tokens = lex("? ?= ?^ ?? ?>").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Question,
                Token::QuestionEq,
                Token::QuestionCaret,
                Token::QuestionQuestion,
                Token::QuestionGt,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_dot_variants() {
        let tokens = lex(". .. ..= .?").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Dot,
                Token::DotDot,
                Token::DotDotEq,
                Token::DotQuestion,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_compound_assignment() {
        let tokens = lex("+= -= *= /= %= **=").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::PlusEq,
                Token::MinusEq,
                Token::StarEq,
                Token::SlashEq,
                Token::PercentEq,
                Token::StarStarEq,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_comment() {
        let tokens = lex("# this is a comment\nx=5").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Eq,
                Token::Int(5),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_hash_length_operator() {
        let tokens = lex("x=#arr").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Eq,
                Token::Hash,
                Token::Ident("arr".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_newline_collapsing() {
        let tokens = lex("x=1\n\n\ny=2").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Eq,
                Token::Int(1),
                Token::Newline,
                Token::Ident("y".into()),
                Token::Eq,
                Token::Int(2),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_semicolons_as_newlines() {
        let tokens = lex("x=1;y=2").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Eq,
                Token::Int(1),
                Token::Newline,
                Token::Ident("y".into()),
                Token::Eq,
                Token::Int(2),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_tuple_access_no_float() {
        // t.1.1 should NOT produce Float(1.1)
        let tokens = lex("t.1.1").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("t".into()),
                Token::Dot,
                Token::Int(1),
                Token::Dot,
                Token::Int(1),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_array_literal() {
        let tokens = lex("[1 2 3]").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LBracket,
                Token::Int(1),
                Token::Int(2),
                Token::Int(3),
                Token::RBracket,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_map_literal() {
        let tokens = lex("{a:1 b:2}").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LBrace,
                Token::Ident("a".into()),
                Token::Colon,
                Token::Int(1),
                Token::Ident("b".into()),
                Token::Colon,
                Token::Int(2),
                Token::RBrace,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_lambda() {
        let tokens = lex(r#"\(x)=x*2"#).unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Backslash,
                Token::LParen,
                Token::Ident("x".into()),
                Token::RParen,
                Token::Eq,
                Token::Ident("x".into()),
                Token::Star,
                Token::Int(2),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_escape_sequences() {
        let tokens = lex(r#""\n\t\\\""{}"#).unwrap();
        // Should not error — the \{ becomes a literal {
        // Actually that's "\n\t\\\"" followed by {}
        // Let's test a simpler case
        let tokens = lex(r#""\n""#).unwrap();
        assert_eq!(tokens, vec![Token::Str("\n".into()), Token::Eof]);
    }

    #[test]
    fn test_go_keyword() {
        let tokens = lex("go{1+2}").unwrap();
        assert_eq!(tokens[0], Token::Go);
    }

    #[test]
    fn test_sel_keyword() {
        let tokens = lex("sel{v=<-c:{v}}").unwrap();
        assert_eq!(tokens[0], Token::Sel);
    }

    #[test]
    fn test_continue_operator() {
        let tokens = lex(">!").unwrap();
        assert_eq!(tokens, vec![Token::GtBang, Token::Eof]);
    }

    #[test]
    fn test_tilde() {
        let tokens = lex("~(i:0..10){p(i)}").unwrap();
        assert_eq!(tokens[0], Token::Tilde);
    }

    #[test]
    fn test_at_import() {
        let tokens = lex(r#"@"math""#).unwrap();
        assert_eq!(
            tokens,
            vec![Token::At, Token::Str("math".into()), Token::Eof,]
        );
    }
}
