use crate::ast::*;
/// Recursive descent parser for the Tok language.
///
/// Converts a flat token stream from tok-lexer into an AST.
/// Handles all 9 phases of the language including:
/// - Expressions with full operator precedence
/// - Functions, lambdas, closures
/// - Control flow: ternary, match, loops
/// - Arrays, maps, tuples
/// - String interpolation
/// - Pipes, filters, reduce
/// - Error handling: ?^, ??, .?
/// - Modules/imports
/// - Concurrency: go, channels, select, pmap
use tok_lexer::Token;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub msg: String,
    pub pos: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "parse error at token {}: {}", self.pos, self.msg)
    }
}

/// Convenience function to parse a token stream into a Program.
pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

/// Lightweight cursor for multi-token lookahead without advancing the parser.
///
/// Wraps a reference to the token slice and a mutable position, providing
/// convenience methods like `skip_newlines()`, `check()`, and `at()` to
/// reduce bounds-checking boilerplate in `is_*` lookahead functions.
struct LookaheadCursor<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> LookaheadCursor<'a> {
    fn new(tokens: &'a [Token], start: usize) -> Self {
        Self { tokens, pos: start }
    }

    /// Current token, or Eof if past end.
    fn at(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    /// Advance by one position.
    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Skip over any newline tokens.
    fn skip_newlines(&mut self) {
        while self.pos < self.tokens.len() && matches!(self.tokens[self.pos], Token::Newline) {
            self.pos += 1;
        }
    }

    /// Check if the current token matches a predicate, advance if true.
    fn eat(&mut self, pred: impl FnOnce(&Token) -> bool) -> bool {
        if self.pos < self.tokens.len() && pred(&self.tokens[self.pos]) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Create a lookahead cursor starting at the given absolute position.
    fn lookahead_from(&self, start: usize) -> LookaheadCursor<'_> {
        LookaheadCursor::new(&self.tokens, start)
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn peek_at(&self, offset: usize) -> &Token {
        self.tokens.get(self.pos + offset).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        if !matches!(tok, Token::Eof) {
            self.pos += 1;
        }
        tok
    }

    fn at(&self, token: &Token) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn expect(&mut self, expected: &Token) -> Result<Token, ParseError> {
        if self.at(expected) {
            Ok(self.advance())
        } else {
            Err(self.error(format!("expected {:?}, found {:?}", expected, self.peek())))
        }
    }

    fn error(&self, msg: String) -> ParseError {
        ParseError { msg, pos: self.pos }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
    }

    /// Check if the next token can start an expression.
    fn can_start_expr(&self) -> bool {
        matches!(
            self.peek(),
            Token::Int(_)
                | Token::Float(_)
                | Token::Str(_)
                | Token::RawStr(_)
                | Token::StringStart(_)
                | Token::True
                | Token::False
                | Token::Nil
                | Token::Ident(_)
                | Token::Func
                | Token::LParen
                | Token::LBracket
                | Token::LBrace
                | Token::Backslash
                | Token::Minus
                | Token::Bang
                | Token::Hash
                | Token::ArrowLeft
                | Token::Tilde
                | Token::At
                | Token::Caret
                | Token::Go
                | Token::Sel
                | Token::QuestionEq
        )
    }

    /// Check if the current position starts a map literal.
    /// We're positioned at `{`. Look ahead (past newlines) for `IDENT:` or `STRING:`.
    /// An empty `{}` is also a map literal.
    fn is_map_literal(&self) -> bool {
        let mut c = self.lookahead_from(self.pos + 1); // past the `{`
        c.skip_newlines();
        // Empty map: `{}`
        if matches!(c.at(), Token::RBrace) {
            return true;
        }
        // Check for `IDENT :` or `STRING :` pattern
        match c.at() {
            Token::Ident(_) | Token::Str(_) | Token::RawStr(_) => {
                c.advance();
                matches!(c.at(), Token::Colon)
            }
            _ => false,
        }
    }

    /// Check if the current position is a tuple destructure: `IDENT IDENT+ =`
    fn is_tuple_destructure(&self) -> bool {
        if !matches!(self.peek(), Token::Ident(_)) {
            return false;
        }
        let mut c = self.lookahead_from(self.pos + 1);
        let mut ident_count = 1;
        while c.eat(|t| matches!(t, Token::Ident(_))) {
            ident_count += 1;
        }
        ident_count >= 2 && matches!(c.at(), Token::Eq)
    }

    /// Check if current position is a map destructure: `{ident ident...} =`
    fn is_map_destructure(&self) -> bool {
        if !matches!(self.peek(), Token::LBrace) {
            return false;
        }
        let mut c = self.lookahead_from(self.pos + 1);
        let mut has_idents = false;
        loop {
            if c.eat(|t| matches!(t, Token::Ident(_))) {
                has_idents = true;
            } else if c.eat(|t| matches!(t, Token::Newline)) {
                // skip
            } else if c.eat(|t| matches!(t, Token::RBrace)) {
                break;
            } else {
                return false;
            }
        }
        has_idents && matches!(c.at(), Token::Eq)
    }

    /// Check if current position is an array destructure: `[ident ..ident] =`
    fn is_array_destructure(&self) -> bool {
        if !matches!(self.peek(), Token::LBracket) {
            return false;
        }
        let mut c = self.lookahead_from(self.pos + 1);
        c.skip_newlines();
        if !c.eat(|t| matches!(t, Token::Ident(_))) {
            return false;
        }
        c.skip_newlines();
        if !c.eat(|t| matches!(t, Token::DotDot)) {
            return false;
        }
        if !c.eat(|t| matches!(t, Token::Ident(_))) {
            return false;
        }
        c.skip_newlines();
        if !c.eat(|t| matches!(t, Token::RBracket)) {
            return false;
        }
        matches!(c.at(), Token::Eq)
    }

    /// Check if current position is assignment: `IDENT = ...` (not `==`)
    fn is_simple_assign(&self) -> bool {
        if !matches!(self.peek(), Token::Ident(_)) {
            return false;
        }
        // Check for : (typed assign) or = (plain assign)
        let next = self.peek_at(1);
        matches!(next, Token::Eq) || matches!(next, Token::Colon)
    }

    /// Check if current position is a compound assignment: `IDENT op=`
    fn is_compound_assign(&self) -> bool {
        if !matches!(self.peek(), Token::Ident(_)) {
            return false;
        }
        matches!(
            self.peek_at(1),
            Token::PlusEq
                | Token::MinusEq
                | Token::StarEq
                | Token::SlashEq
                | Token::PercentEq
                | Token::StarStarEq
                | Token::LtLtEq
        )
    }

    /// Check if current position is member assign: `IDENT . IDENT =` or deeper chains.
    /// Also need to handle `IDENT . IDENT op=` for compound member assign.
    /// And `IDENT [ expr ] =` for index assign.
    fn is_member_or_index_assign(&self) -> bool {
        if !matches!(self.peek(), Token::Ident(_)) {
            return false;
        }
        // Walk past the postfix chain (`.field`, `[expr]`) to see if it
        // ends with `=` or a compound assignment operator. This avoids the
        // expensive parse_postfix + backtrack when the expression is not
        // actually an assignment.
        let mut la = self.lookahead_from(self.pos + 1); // skip initial Ident
        let mut saw_member_or_index = false;
        loop {
            match la.at() {
                Token::Dot => {
                    saw_member_or_index = true;
                    la.advance(); // skip `.`
                    // skip field name (Ident or Int for tuple index)
                    if matches!(la.at(), Token::Ident(_) | Token::Int(_)) {
                        la.advance();
                    } else {
                        return false;
                    }
                }
                Token::LBracket => {
                    saw_member_or_index = true;
                    la.advance(); // skip `[`
                    let mut depth = 1u32;
                    while depth > 0 {
                        match la.at() {
                            Token::LBracket => { depth += 1; la.advance(); }
                            Token::RBracket => { depth -= 1; la.advance(); }
                            Token::Eof => return false,
                            _ => la.advance(),
                        }
                    }
                }
                _ => break,
            }
        }
        if !saw_member_or_index {
            return false;
        }
        matches!(
            la.at(),
            Token::Eq
                | Token::PlusEq
                | Token::MinusEq
                | Token::StarEq
                | Token::SlashEq
                | Token::PercentEq
                | Token::StarStarEq
                | Token::LtLtEq
        )
    }

    /// Try to parse the left-hand side of a member/index assignment and determine
    /// what kind of assignment it is. Returns None if it's not an assignment.
    fn try_member_index_assign(&mut self) -> Result<Option<Stmt>, ParseError> {
        // Save position
        let saved_pos = self.pos;

        // Parse the target expression (which may be `ident`, `ident.field`, `ident[idx]`, etc.)
        // We parse a postfix expression, then check if it's followed by `=` or `op=`.
        let expr = self.parse_postfix()?;

        // Check what follows
        match self.peek() {
            Token::Eq => {
                self.advance(); // consume =
                let value = self.parse_expr()?;
                match expr {
                    Expr::Member {
                        expr: target,
                        field,
                    } => Ok(Some(Stmt::MemberAssign {
                        target: *target,
                        field,
                        value,
                    })),
                    Expr::OptionalChain {
                        expr: target,
                        field,
                    } => Ok(Some(Stmt::MemberAssign {
                        target: *target,
                        field,
                        value,
                    })),
                    Expr::Index {
                        expr: target,
                        index,
                    } => Ok(Some(Stmt::IndexAssign {
                        target: *target,
                        index: *index,
                        value,
                    })),
                    _ => {
                        // Not a valid assignment target, restore
                        self.pos = saved_pos;
                        Ok(None)
                    }
                }
            }
            Token::PlusEq
            | Token::MinusEq
            | Token::StarEq
            | Token::SlashEq
            | Token::PercentEq
            | Token::StarStarEq
            | Token::LtLtEq => {
                let tok = self.advance();
                let op = self.compound_op_to_binop(&tok);
                let value = self.parse_expr()?;
                match expr {
                    Expr::Member {
                        expr: target,
                        field,
                    } => Ok(Some(Stmt::CompoundMemberAssign {
                        target: *target,
                        field,
                        op,
                        value,
                    })),
                    Expr::Index {
                        expr: target,
                        index,
                    } => Ok(Some(Stmt::CompoundIndexAssign {
                        target: *target,
                        index: *index,
                        op,
                        value,
                    })),
                    _ => {
                        self.pos = saved_pos;
                        Ok(None)
                    }
                }
            }
            _ => {
                // Not an assignment, restore and return None
                self.pos = saved_pos;
                Ok(None)
            }
        }
    }

    fn compound_op_to_binop(&self, tok: &Token) -> BinOp {
        match tok {
            Token::PlusEq => BinOp::Add,
            Token::MinusEq => BinOp::Sub,
            Token::StarEq => BinOp::Mul,
            Token::SlashEq => BinOp::Div,
            Token::PercentEq => BinOp::Mod,
            Token::StarStarEq => BinOp::Pow,
            Token::LtLtEq => BinOp::Append,
            _ => unreachable!(),
        }
    }

    // ---------------------------------------------------------------
    // Top-level
    // ---------------------------------------------------------------

    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !self.at_eof() {
            let stmt = self.parse_stmt()?;
            stmts.push(stmt);
            self.skip_newlines();
        }
        Ok(stmts)
    }

    // ---------------------------------------------------------------
    // Statement parsing
    // ---------------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.skip_newlines();

        match self.peek().clone() {
            // Function declaration: f name(...)
            Token::Func => {
                return self.parse_func_decl();
            }

            // Bare import: @"path" (at statement level, not assigned to anything)
            Token::At => {
                // Check if this is a bare import: @"string" at statement level
                // (not @"path" used as an expression in assignment)
                // We need to check if this is `@"path"` by itself (not `name = @"path"`)
                // Since we reach here, At is the first token of the statement.
                // A bare import is: @"path" followed by newline/eof/;
                // But `(@"path").field` would parse as expression. Let's check.
                // If it's `@"str"` followed by a newline/eof, it's a bare import.
                if matches!(self.peek_at(1), Token::Str(_) | Token::RawStr(_)) {
                    let saved = self.pos;
                    self.advance(); // consume @
                    let path = match self.advance() {
                        Token::Str(s) | Token::RawStr(s) => s,
                        _ => unreachable!(),
                    };
                    // If followed by newline, eof, or nothing else meaningful, it's bare import
                    if matches!(self.peek(), Token::Newline | Token::Eof) || !self.can_start_expr()
                    {
                        return Ok(Stmt::Import(path));
                    }
                    // Otherwise restore and parse as expression
                    self.pos = saved;
                }
            }

            // Early return: ^expr or bare ^
            Token::Caret => {
                self.advance();
                if self.can_start_expr() {
                    let expr = self.parse_expr()?;
                    return Ok(Stmt::Return(Some(expr)));
                } else {
                    return Ok(Stmt::Return(None));
                }
            }

            // Break: !
            Token::Bang => {
                // `!` at statement start = break (but could also be `!expr`)
                // If next token can start an expression, it's `!expr` (unary not).
                // But `!` at statement start by itself = break.
                // Actually, `!` is the break keyword at statement start.
                // From the test files: `!` is break only in loops.
                // `!expr` is boolean not. Let's look at the spec...
                // The break is `!` as a standalone statement (not followed by something
                // that could be operand of `!` in a useful statement way).
                // Actually, looking at AST: Stmt::Break is `!`. And `!expr` would be
                // an expression stmt with UnaryOp::Not.
                // We should check: is `!` followed by a newline/eof/rbrace? Then break.
                // Otherwise it's `!expr`.
                if matches!(self.peek_at(1), Token::Newline | Token::Eof | Token::RBrace) {
                    self.advance();
                    return Ok(Stmt::Break);
                }
                // Otherwise fall through to expression parsing (it's `!expr`)
            }

            // Continue: >!
            Token::GtBang => {
                self.advance();
                return Ok(Stmt::Continue);
            }

            // Tuple destructure: a b c = expr
            Token::Ident(_) if self.is_tuple_destructure() => {
                return self.parse_tuple_destructure();
            }

            // Map destructure: {a b c} = expr
            Token::LBrace if self.is_map_destructure() => {
                return self.parse_map_destructure();
            }

            // Array destructure: [h ..t] = expr
            Token::LBracket if self.is_array_destructure() => {
                return self.parse_array_destructure();
            }

            // Compound assignment: name op= expr
            Token::Ident(_) if self.is_compound_assign() => {
                return self.parse_compound_assign();
            }

            // Member/index assignment: name.field = expr, name[idx] = expr
            // or compound variants: name.field op= expr, name[idx] op= expr
            Token::Ident(_) if self.is_member_or_index_assign() => {
                if let Some(stmt) = self.try_member_index_assign()? {
                    return Ok(stmt);
                }
                // If it wasn't actually an assignment, fall through to expression parsing
            }

            // Simple assignment: name = expr or name:type = expr
            Token::Ident(_) if self.is_simple_assign() => {
                return self.parse_assign();
            }

            _ => {}
        }

        // Default: expression statement
        let expr = self.parse_expr()?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_func_decl(&mut self) -> Result<Stmt, ParseError> {
        self.expect(&Token::Func)?; // consume `f`
        let name = match self.advance() {
            Token::Ident(s) => s,
            other => return Err(self.error(format!("expected function name, found {:?}", other))),
        };
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;

        // Optional return type
        let ret_type = if matches!(self.peek(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        // Body: `= expr` or `{ stmts }`
        let body = if matches!(self.peek(), Token::Eq) {
            self.advance();
            let expr = self.parse_expr()?;
            FuncBody::Expr(Box::new(expr))
        } else if matches!(self.peek(), Token::LBrace) {
            FuncBody::Block(self.parse_block()?)
        } else {
            return Err(self.error(format!(
                "expected '=' or '{{' after function params, found {:?}",
                self.peek()
            )));
        };

        Ok(Stmt::FuncDecl {
            name,
            params,
            ret_type,
            body,
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RParen | Token::Eof) {
            let variadic = if matches!(self.peek(), Token::DotDot) {
                self.advance();
                true
            } else {
                false
            };

            let name = match self.advance() {
                Token::Ident(s) => s,
                other => {
                    return Err(self.error(format!("expected parameter name, found {:?}", other)))
                }
            };

            // Optional type annotation
            let ty = if matches!(self.peek(), Token::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            // Optional default value
            let default = if matches!(self.peek(), Token::Eq) {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };

            params.push(Param {
                name,
                ty,
                default,
                variadic,
            });
            self.skip_newlines();
        }
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            let stmt = self.parse_stmt()?;
            stmts.push(stmt);
            self.skip_newlines();
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    fn parse_assign(&mut self) -> Result<Stmt, ParseError> {
        let name = match self.advance() {
            Token::Ident(s) => s,
            _ => unreachable!(),
        };

        // Check for type annotation
        let ty = if matches!(self.peek(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;

        Ok(Stmt::Assign { name, ty, value })
    }

    fn parse_compound_assign(&mut self) -> Result<Stmt, ParseError> {
        let name = match self.advance() {
            Token::Ident(s) => s,
            _ => unreachable!(),
        };
        let op_tok = self.advance();
        let op = self.compound_op_to_binop(&op_tok);
        let value = self.parse_expr()?;
        Ok(Stmt::CompoundAssign { name, op, value })
    }

    fn parse_tuple_destructure(&mut self) -> Result<Stmt, ParseError> {
        let mut names = Vec::new();
        // Collect all identifiers until `=`
        while matches!(self.peek(), Token::Ident(_)) {
            match self.advance() {
                Token::Ident(s) => names.push(s),
                _ => unreachable!(),
            }
        }
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::TupleDestructure { names, value })
    }

    fn parse_map_destructure(&mut self) -> Result<Stmt, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut names = Vec::new();
        self.skip_newlines();
        while matches!(self.peek(), Token::Ident(_)) {
            match self.advance() {
                Token::Ident(s) => names.push(s),
                _ => unreachable!(),
            }
            self.skip_newlines();
        }
        self.expect(&Token::RBrace)?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::MapDestructure { names, value })
    }

    fn parse_array_destructure(&mut self) -> Result<Stmt, ParseError> {
        self.expect(&Token::LBracket)?;
        self.skip_newlines();
        let head = match self.advance() {
            Token::Ident(s) => s,
            other => {
                return Err(self.error(format!(
                    "expected identifier in array destructure, found {:?}",
                    other
                )))
            }
        };
        self.skip_newlines();
        self.expect(&Token::DotDot)?;
        let tail = match self.advance() {
            Token::Ident(s) => s,
            other => {
                return Err(self.error(format!(
                    "expected identifier after .. in array destructure, found {:?}",
                    other
                )))
            }
        };
        self.skip_newlines();
        self.expect(&Token::RBracket)?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::ArrayDestructure { head, tail, value })
    }

    // ---------------------------------------------------------------
    // Type parsing
    // ---------------------------------------------------------------

    fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        let base = self.parse_type_base()?;
        // Postfix `?` for optional
        if matches!(self.peek(), Token::Question) {
            self.advance();
            return Ok(TypeExpr::Optional(Box::new(base)));
        }
        Ok(base)
    }

    fn parse_type_base(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            Token::Ident(ref s) => {
                let s = s.clone();
                self.advance();
                match s.as_str() {
                    "i" => Ok(TypeExpr::Prim(PrimType::Int)),
                    "f" => Ok(TypeExpr::Prim(PrimType::Float)),
                    "s" => Ok(TypeExpr::Prim(PrimType::Str)),
                    "b" => Ok(TypeExpr::Prim(PrimType::Bool)),
                    "a" => Ok(TypeExpr::Prim(PrimType::Any)),
                    "R" => {
                        // Result type: R(T)
                        self.expect(&Token::LParen)?;
                        let inner = self.parse_type()?;
                        self.expect(&Token::RParen)?;
                        Ok(TypeExpr::Result(Box::new(inner)))
                    }
                    "C" => {
                        // Channel type: C(T)
                        self.expect(&Token::LParen)?;
                        let inner = self.parse_type()?;
                        self.expect(&Token::RParen)?;
                        Ok(TypeExpr::Channel(Box::new(inner)))
                    }
                    "H" => {
                        // Handle type: H(T)
                        self.expect(&Token::LParen)?;
                        let inner = self.parse_type()?;
                        self.expect(&Token::RParen)?;
                        Ok(TypeExpr::Handle(Box::new(inner)))
                    }
                    _ if s.len() == 1 && s.chars().next().unwrap().is_ascii_uppercase() => {
                        // Type variable (single uppercase letter, not N which is Nil)
                        Ok(TypeExpr::Var(s))
                    }
                    _ => Err(self.error(format!("unknown type: {}", s))),
                }
            }
            Token::Nil => {
                self.advance();
                Ok(TypeExpr::Prim(PrimType::Nil))
            }
            Token::LBracket => {
                // Array type: [T]
                self.advance();
                let inner = self.parse_type()?;
                self.expect(&Token::RBracket)?;
                Ok(TypeExpr::Array(Box::new(inner)))
            }
            Token::LBrace => {
                // Map type: {T}
                self.advance();
                let inner = self.parse_type()?;
                self.expect(&Token::RBrace)?;
                Ok(TypeExpr::Map(Box::new(inner)))
            }
            Token::LParen => {
                // Tuple type: (T U V)
                self.advance();
                let mut types = Vec::new();
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    types.push(self.parse_type()?);
                }
                self.expect(&Token::RParen)?;
                Ok(TypeExpr::Tuple(types))
            }
            Token::Backslash => {
                // Function type: \(T U):R
                self.advance();
                self.expect(&Token::LParen)?;
                let mut param_types = Vec::new();
                while !matches!(self.peek(), Token::RParen | Token::Eof) {
                    param_types.push(self.parse_type()?);
                }
                self.expect(&Token::RParen)?;
                self.expect(&Token::Colon)?;
                let ret = self.parse_type()?;
                Ok(TypeExpr::Func(param_types, Box::new(ret)))
            }
            _ => Err(self.error(format!("expected type, found {:?}", self.peek()))),
        }
    }

    // ---------------------------------------------------------------
    // Expression parsing — precedence climbing
    // ---------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_ternary()
    }

    /// Prec 0: Ternary `cond ? then : else`, pipes `|>`, filter `?>`, reduce `/>`,
    /// nil coalesce `??`, conditional return `cond?^val`, error propagation `expr?^`
    fn parse_ternary(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_or()?;

        loop {
            match self.peek().clone() {
                Token::Question => {
                    self.advance();
                    // Special case: cond?! (conditional break) and cond?>! (conditional continue)
                    let then_expr = if matches!(self.peek(), Token::Bang)
                        && matches!(
                            self.peek_at(1),
                            Token::Newline | Token::Eof | Token::RBrace | Token::Colon
                        ) {
                        self.advance(); // consume !
                        Expr::Break
                    } else if matches!(self.peek(), Token::GtBang)
                        && matches!(
                            self.peek_at(1),
                            Token::Newline | Token::Eof | Token::RBrace | Token::Colon
                        )
                    {
                        self.advance(); // consume >!
                        Expr::Continue
                    } else {
                        // Ternary: expr ? then : else
                        // or expr ? then (without else)
                        self.parse_ternary()?
                    };
                    let else_expr = if matches!(self.peek(), Token::Colon) {
                        self.advance();
                        Some(Box::new(self.parse_ternary()?))
                    } else {
                        None
                    };
                    expr = Expr::Ternary {
                        cond: Box::new(expr),
                        then_expr: Box::new(then_expr),
                        else_expr,
                    };
                }
                Token::QuestionCaret => {
                    self.advance();
                    // Dual meaning:
                    // - cond?^expr (conditional return) — if next can start expr
                    // - expr?^ (error propagation) — if next cannot start expr
                    if self.can_start_expr() {
                        let value = self.parse_ternary()?;
                        expr = Expr::ConditionalReturn {
                            cond: Box::new(expr),
                            value: Box::new(value),
                        };
                    } else {
                        expr = Expr::ErrorPropagate(Box::new(expr));
                    }
                }
                Token::QuestionQuestion => {
                    self.advance();
                    // Right-associative nil coalesce
                    let right = self.parse_ternary()?;
                    expr = Expr::NilCoalesce {
                        left: Box::new(expr),
                        right: Box::new(right),
                    };
                }
                Token::QuestionEq => {
                    self.advance();
                    // Value match: expr?={ arms }
                    self.expect(&Token::LBrace)?;
                    let arms = self.parse_match_arms()?;
                    self.expect(&Token::RBrace)?;
                    expr = Expr::Match {
                        subject: Some(Box::new(expr)),
                        arms,
                    };
                }
                Token::PipeGt => {
                    self.advance();
                    let right = self.parse_or()?;
                    expr = Expr::Pipeline {
                        left: Box::new(expr),
                        right: Box::new(right),
                    };
                }
                Token::QuestionGt => {
                    // Special case: cond?>! is conditional continue (? then >!)
                    if matches!(self.peek_at(1), Token::Bang)
                        && matches!(
                            self.peek_at(2),
                            Token::Newline | Token::Eof | Token::RBrace | Token::Colon
                        )
                    {
                        self.advance(); // consume ?>
                        self.advance(); // consume !
                        let then_expr = Expr::Continue;
                        let else_expr = if matches!(self.peek(), Token::Colon) {
                            self.advance();
                            Some(Box::new(self.parse_ternary()?))
                        } else {
                            None
                        };
                        expr = Expr::Ternary {
                            cond: Box::new(expr),
                            then_expr: Box::new(then_expr),
                            else_expr,
                        };
                    } else {
                        self.advance();
                        // Filter: expr ?> func
                        // Parse at parse_or level so `/>` isn't consumed
                        let pred = self.parse_or()?;
                        expr = Expr::Filter {
                            expr: Box::new(expr),
                            pred: Box::new(pred),
                        };
                        // After filter, allow chaining with />
                        // (handled by the loop continuing)
                    }
                }
                Token::SlashGt => {
                    self.advance();
                    // Reduce: expr /> [init] func
                    // Optional init value: if next token can start an expression and
                    // is not a backslash (lambda), it might be init.
                    // Actually, the init is present when there's a value before the lambda.
                    // Pattern: `arr/>0 \(a x)=a+x` or `arr/>\(a x)=a+x`
                    let (init, func) = self.parse_reduce_args()?;
                    expr = Expr::Reduce {
                        expr: Box::new(expr),
                        init: init.map(Box::new),
                        func: Box::new(func),
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_reduce_args(&mut self) -> Result<(Option<Expr>, Expr), ParseError> {
        // Check if the next thing is a lambda (\) — then no init
        if matches!(self.peek(), Token::Backslash) {
            let func = self.parse_or()?;
            return Ok((None, func));
        }
        // Otherwise the next thing is init, then the function
        // But we need to be careful: the init could be a complex expression.
        // In practice from the test files, init is a simple literal like `0` or `10`.
        // The pattern is: `/>init func` where init is a simple expression
        // and func is a lambda or identifier.
        // Let's parse the first thing as init, then the next as func.
        let init = self.parse_or()?;
        let func = self.parse_or()?;
        Ok((Some(init), func))
    }

    /// Prec 1: Logical OR `|`
    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_and()?;
        while matches!(self.peek(), Token::Pipe) {
            self.advance();
            let right = self.parse_and()?;
            expr = Expr::BinOp {
                op: BinOp::Or,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 2: Logical AND `&`
    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_comparison()?;
        while matches!(self.peek(), Token::Amp) {
            self.advance();
            let right = self.parse_comparison()?;
            expr = Expr::BinOp {
                op: BinOp::And,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 3: Comparison `== != > < >= <=`
    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_bitor()?;
        loop {
            let op = match self.peek() {
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Neq,
                Token::Gt => BinOp::Gt,
                Token::Lt => BinOp::Lt,
                Token::GtEq => BinOp::GtEq,
                Token::LtEq => BinOp::LtEq,
                _ => break,
            };
            self.advance();
            let right = self.parse_bitor()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 4: Bitwise OR `||`
    fn parse_bitor(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_bitxor()?;
        while matches!(self.peek(), Token::PipePipe) {
            self.advance();
            let right = self.parse_bitxor()?;
            expr = Expr::BinOp {
                op: BinOp::BitOr,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 5: Bitwise XOR `^^`
    fn parse_bitxor(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_bitand()?;
        while matches!(self.peek(), Token::CaretCaret) {
            self.advance();
            let right = self.parse_bitand()?;
            expr = Expr::BinOp {
                op: BinOp::BitXor,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 6: Bitwise AND `&&`
    fn parse_bitand(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_shift()?;
        while matches!(self.peek(), Token::AmpAmp) {
            self.advance();
            let right = self.parse_shift()?;
            expr = Expr::BinOp {
                op: BinOp::BitAnd,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 7: Append/Shift `<< >>`
    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_range()?;
        loop {
            let op = match self.peek() {
                Token::LtLt => BinOp::Append,
                Token::GtGt => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_range()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 8: Range `.. ..=` (non-associative)
    fn parse_range(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_additive()?;
        match self.peek() {
            Token::DotDot => {
                self.advance();
                let end = self.parse_additive()?;
                Ok(Expr::Range {
                    start: Box::new(expr),
                    end: Box::new(end),
                    inclusive: false,
                })
            }
            Token::DotDotEq => {
                self.advance();
                let end = self.parse_additive()?;
                Ok(Expr::Range {
                    start: Box::new(expr),
                    end: Box::new(end),
                    inclusive: true,
                })
            }
            _ => Ok(expr),
        }
    }

    /// Prec 9: Additive `+ -`
    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 10: Multiplicative `* / %`
    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_power()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_power()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    /// Prec 11: Power `**` (right-associative)
    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_unary()?;
        if matches!(self.peek(), Token::StarStar) {
            self.advance();
            let right = self.parse_power()?; // right-recursive for right-associativity
            Ok(Expr::BinOp {
                op: BinOp::Pow,
                left: Box::new(expr),
                right: Box::new(right),
            })
        } else {
            Ok(expr)
        }
    }

    /// Prec 12: Unary prefix `! - # <-`
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Token::Bang => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            Token::Hash => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Length(Box::new(expr)))
            }
            Token::ArrowLeft => {
                // Prefix receive: <-expr
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Receive(Box::new(expr)))
            }
            _ => self.parse_postfix(),
        }
    }

    /// Prec 13-14: Postfix `.field`, `.?field`, `[index]`, `(args)`, `<-val` (send)
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek().clone() {
                Token::Dot => {
                    self.advance();
                    match self.peek().clone() {
                        Token::Ident(field) => {
                            self.advance();
                            expr = Expr::Member {
                                expr: Box::new(expr),
                                field,
                            };
                        }
                        Token::Int(n) => {
                            // Tuple index: expr.0, expr.1
                            self.advance();
                            expr = Expr::Member {
                                expr: Box::new(expr),
                                field: n.to_string(),
                            };
                        }
                        _ => {
                            return Err(self.error(format!(
                                "expected field name or index after '.', found {:?}",
                                self.peek()
                            )));
                        }
                    }
                }
                Token::DotQuestion => {
                    self.advance();
                    match self.peek().clone() {
                        Token::Ident(field) => {
                            self.advance();
                            expr = Expr::OptionalChain {
                                expr: Box::new(expr),
                                field,
                            };
                        }
                        Token::Int(n) => {
                            self.advance();
                            expr = Expr::OptionalChain {
                                expr: Box::new(expr),
                                field: n.to_string(),
                            };
                        }
                        _ => {
                            return Err(self.error(format!(
                                "expected field name after '.?', found {:?}",
                                self.peek()
                            )));
                        }
                    }
                }
                Token::LBracket => {
                    // GOTCHA: If current expr is Array, don't chain — it would be
                    // `[1 2][3 4]` being misinterpreted as index access.
                    if matches!(expr, Expr::Array(_)) {
                        break;
                    }
                    self.advance();
                    self.skip_newlines();
                    let index = self.parse_expr()?;
                    self.skip_newlines();
                    self.expect(&Token::RBracket)?;
                    expr = Expr::Index {
                        expr: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Token::LParen => {
                    // GOTCHA: If current expr is Tuple, don't chain — it would be
                    // `(1 N)(2 N)` being misinterpreted as function call.
                    if matches!(expr, Expr::Tuple(_)) {
                        break;
                    }
                    self.advance();
                    let args = self.parse_call_args()?;
                    self.expect(&Token::RParen)?;
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                    };
                }
                Token::ArrowLeft => {
                    // Infix send: chan <- val
                    self.advance();
                    let value = self.parse_expr()?;
                    expr = Expr::Send {
                        chan: Box::new(expr),
                        value: Box::new(value),
                    };
                    // Send consumes the rest, so break out
                    break;
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RParen | Token::Eof) {
            args.push(self.parse_expr()?);
            self.skip_newlines();
        }
        Ok(args)
    }

    /// Primary expressions: literals, identifiers, grouping, arrays, maps, blocks,
    /// lambdas, match, loop, go, select, import, string interpolation.
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Token::Int(n) => {
                self.advance();
                Ok(Expr::Int(n))
            }
            Token::Float(n) => {
                self.advance();
                Ok(Expr::Float(n))
            }
            Token::Str(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Token::RawStr(s) => {
                self.advance();
                Ok(Expr::Str(s))
            }
            Token::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Token::Nil => {
                self.advance();
                Ok(Expr::Nil)
            }
            Token::Ident(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::Ident(s))
            }
            Token::StringStart(_) => self.parse_interpolated_string(),
            Token::LParen => self.parse_paren_expr(),
            Token::LBracket => self.parse_array_literal(),
            Token::LBrace => {
                if self.is_map_literal() {
                    self.parse_map_literal()
                } else {
                    // Block expression
                    let stmts = self.parse_block()?;
                    Ok(Expr::Block(stmts))
                }
            }
            Token::Backslash => self.parse_lambda(),
            Token::Tilde => self.parse_loop(),
            Token::QuestionEq => {
                // Standalone match: ?={ arms }
                self.advance();
                self.expect(&Token::LBrace)?;
                let arms = self.parse_match_arms()?;
                self.expect(&Token::RBrace)?;
                Ok(Expr::Match {
                    subject: None,
                    arms,
                })
            }
            Token::At => {
                // Import expression: @"path"
                self.advance();
                let path = match self.advance() {
                    Token::Str(s) | Token::RawStr(s) => s,
                    other => {
                        return Err(
                            self.error(format!("expected string after @, found {:?}", other))
                        )
                    }
                };
                Ok(Expr::Import(path))
            }
            Token::Go => {
                // go { expr }
                self.advance();
                self.expect(&Token::LBrace)?;
                let mut stmts = Vec::new();
                self.skip_newlines();
                while !matches!(self.peek(), Token::RBrace | Token::Eof) {
                    stmts.push(self.parse_stmt()?);
                    self.skip_newlines();
                }
                self.expect(&Token::RBrace)?;
                // If single expression statement, unwrap it
                let inner = if stmts.len() == 1 {
                    match stmts.into_iter().next().unwrap() {
                        Stmt::Expr(e) => e,
                        other => Expr::Block(vec![other]),
                    }
                } else {
                    Expr::Block(stmts)
                };
                Ok(Expr::Go(Box::new(inner)))
            }
            Token::Sel => self.parse_select(),
            Token::Caret => {
                // Early return as expression: ^expr
                self.advance();
                if self.can_start_expr() {
                    let e = self.parse_expr()?;
                    Ok(Expr::Return(Some(Box::new(e))))
                } else {
                    Ok(Expr::Return(None))
                }
            }
            Token::Func => {
                // `f` as identifier (type name or variable) — the lexer only emits
                // Token::Func when followed by an identifier (function declaration).
                // But at expression level, this shouldn't happen... unless used in type context.
                // Actually, looking at it: when `f` appears in expression context (like `type_of(...) == "f"`),
                // the lexer emits Token::Ident("f"), not Token::Func.
                // Token::Func only appears when lexer sees `f ident`.
                // So reaching here means there's a `f name(...)` in expression context,
                // which isn't valid. Let's error.
                Err(self.error("unexpected function declaration in expression context".into()))
            }
            _ => Err(self.error(format!("unexpected token: {:?}", self.peek()))),
        }
    }

    // ---------------------------------------------------------------
    // Specific expression forms
    // ---------------------------------------------------------------

    /// Parse parenthesized expression or tuple.
    /// `(expr)` — grouping
    /// `(expr expr ...)` — tuple
    fn parse_paren_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::LParen)?;
        self.skip_newlines();

        if matches!(self.peek(), Token::RParen) {
            // Empty tuple ()
            self.advance();
            return Ok(Expr::Tuple(Vec::new()));
        }

        let first = self.parse_expr()?;
        self.skip_newlines();

        if matches!(self.peek(), Token::RParen) {
            // Single expression in parens: grouping, not tuple
            self.advance();
            return Ok(first);
        }

        // Multiple expressions: tuple
        let mut elements = vec![first];
        while !matches!(self.peek(), Token::RParen | Token::Eof) {
            elements.push(self.parse_expr()?);
            self.skip_newlines();
        }
        self.expect(&Token::RParen)?;
        Ok(Expr::Tuple(elements))
    }

    /// Parse array literal: `[elem1 elem2 ...]`
    /// Elements can be spread: `[..expr ...]`
    fn parse_array_literal(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::LBracket)?;
        let mut elements = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RBracket | Token::Eof) {
            if matches!(self.peek(), Token::DotDot) {
                self.advance();
                // Parse at additive level to avoid consuming `..` for the next spread
                let spread_expr = self.parse_postfix()?;
                elements.push(Expr::Spread(Box::new(spread_expr)));
            } else {
                // Parse at additive level to avoid consuming `..` as range operator.
                // In arrays, `..` means spread, not range.
                // Use parentheses for ranges in arrays: [(0..5)]
                elements.push(self.parse_additive()?);
            }
            self.skip_newlines();
        }
        self.expect(&Token::RBracket)?;
        Ok(Expr::Array(elements))
    }

    /// Parse map literal: `{key:val key:val ...}` or `{}`
    fn parse_map_literal(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            let key = match self.advance() {
                Token::Ident(s) => MapKey::Ident(s),
                Token::Str(s) => MapKey::Str(s),
                Token::RawStr(s) => MapKey::Str(s),
                other => return Err(self.error(format!("expected map key, found {:?}", other))),
            };
            self.expect(&Token::Colon)?;
            let value = self.parse_expr()?;
            entries.push((key, value));
            self.skip_newlines();
        }
        self.expect(&Token::RBrace)?;
        Ok(Expr::Map(entries))
    }

    /// Parse lambda: `\(params):rettype = expr` or `\(params):rettype { block }`
    /// or shorthand: `\= expr` (no params) — wait, looking at test files, the shorthand
    /// doesn't seem to exist. Let's support `\(params)=expr` and `\(params){block}`.
    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::Backslash)?;

        // Check for shorthand: \= expr (no params)
        if matches!(self.peek(), Token::Eq) {
            self.advance();
            let expr = self.parse_expr()?;
            return Ok(Expr::Lambda {
                params: Vec::new(),
                ret_type: None,
                body: FuncBody::Expr(Box::new(expr)),
            });
        }

        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(&Token::RParen)?;

        // Optional return type
        let ret_type = if matches!(self.peek(), Token::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        // Body — parse at parse_or level so `?>`, `/>`, `|>` etc. are not consumed
        // (they belong to the outer expression that contains the lambda)
        let body = if matches!(self.peek(), Token::Eq) {
            self.advance();
            FuncBody::Expr(Box::new(self.parse_or()?))
        } else if matches!(self.peek(), Token::LBrace) {
            FuncBody::Block(self.parse_block()?)
        } else {
            return Err(self.error(format!(
                "expected '=' or '{{' in lambda, found {:?}",
                self.peek()
            )));
        };

        Ok(Expr::Lambda {
            params,
            ret_type,
            body,
        })
    }

    /// Parse string interpolation: StringStart expr (StringMid expr)* StringEnd
    fn parse_interpolated_string(&mut self) -> Result<Expr, ParseError> {
        let mut parts = Vec::new();
        match self.advance() {
            Token::StringStart(text) => {
                if !text.is_empty() {
                    parts.push(InterpPart::Lit(text));
                }
            }
            _ => unreachable!(),
        }

        // Parse first interpolated expression
        let expr = self.parse_expr()?;
        parts.push(InterpPart::Expr(expr));

        // Parse remaining segments
        loop {
            match self.peek().clone() {
                Token::StringMid(text) => {
                    self.advance();
                    if !text.is_empty() {
                        parts.push(InterpPart::Lit(text));
                    }
                    let expr = self.parse_expr()?;
                    parts.push(InterpPart::Expr(expr));
                }
                Token::StringEnd(text) => {
                    self.advance();
                    if !text.is_empty() {
                        parts.push(InterpPart::Lit(text));
                    }
                    break;
                }
                _ => {
                    return Err(self.error(format!(
                        "expected StringMid or StringEnd, found {:?}",
                        self.peek()
                    )));
                }
            }
        }

        Ok(Expr::Interp(parts))
    }

    /// Parse loop: `~(clause){body}` or `~(clause)=expr` or `~{body}`
    fn parse_loop(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::Tilde)?;

        // Infinite loop: ~{body}
        if matches!(self.peek(), Token::LBrace) {
            let body = self.parse_block()?;
            return Ok(Expr::Loop {
                clause: Box::new(LoopClause::Infinite),
                body: Box::new(LoopBody::Block(body)),
            });
        }

        // Loop with clause: ~(clause){body} or ~(clause)=expr
        self.expect(&Token::LParen)?;
        let clause = self.parse_loop_clause()?;
        self.expect(&Token::RParen)?;

        // Body: {block} or =expr (collect)
        let body = if matches!(self.peek(), Token::Eq) {
            self.advance();
            let expr = self.parse_expr()?;
            LoopBody::Collect(expr)
        } else if matches!(self.peek(), Token::LBrace) {
            let stmts = self.parse_block()?;
            LoopBody::Block(stmts)
        } else {
            return Err(self.error(format!(
                "expected '{{' or '=' after loop clause, found {:?}",
                self.peek()
            )));
        };

        Ok(Expr::Loop {
            clause: Box::new(clause),
            body: Box::new(body),
        })
    }

    /// Parse loop clause inside `~(...)`:
    /// - While: `cond` (expression without `:`)
    /// - ForRange: `var:range_expr`
    /// - ForEach: `var:iter_expr`
    /// - ForEachIndexed: `idx_var val_var:iter_expr`
    fn parse_loop_clause(&mut self) -> Result<LoopClause, ParseError> {
        // Check if it starts with `ident :` (foreach/range) or `ident ident :` (indexed)
        // or is just an expression (while).
        if matches!(self.peek(), Token::Ident(_)) {
            // Look ahead to distinguish:
            // `ident : expr` → foreach/range
            // `ident ident : expr` → indexed foreach
            // `ident <other>` → while (expression starting with ident)
            if matches!(self.peek_at(1), Token::Colon) {
                // ForRange or ForEach: `var : expr`
                let var = match self.advance() {
                    Token::Ident(s) => s,
                    _ => unreachable!(),
                };
                self.advance(); // consume :
                let iter_expr = self.parse_expr()?;
                // Determine if it's a range or foreach based on the expression type
                match iter_expr {
                    Expr::Range { .. } => Ok(LoopClause::ForRange {
                        var,
                        range: iter_expr,
                    }),
                    _ => Ok(LoopClause::ForEach {
                        var,
                        iter: iter_expr,
                    }),
                }
            } else if matches!(self.peek_at(1), Token::Ident(_))
                && matches!(self.peek_at(2), Token::Colon)
            {
                // ForEachIndexed: `idx val : expr`
                let idx_var = match self.advance() {
                    Token::Ident(s) => s,
                    _ => unreachable!(),
                };
                let val_var = match self.advance() {
                    Token::Ident(s) => s,
                    _ => unreachable!(),
                };
                self.advance(); // consume :
                let iter_expr = self.parse_expr()?;
                Ok(LoopClause::ForEachIndexed {
                    idx_var,
                    val_var,
                    iter: iter_expr,
                })
            } else {
                // While: just an expression
                let cond = self.parse_expr()?;
                Ok(LoopClause::While(cond))
            }
        } else {
            // While: expression
            let cond = self.parse_expr()?;
            Ok(LoopClause::While(cond))
        }
    }

    /// Parse match arms: `pattern : body ; pattern : body ; ...`
    /// Each arm is `pattern : expr` or `pattern : {block}`
    /// `_` is the wildcard/default pattern.
    fn parse_match_arms(&mut self) -> Result<Vec<MatchArm>, ParseError> {
        let mut arms = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            let pattern = self.parse_pattern()?;
            self.expect(&Token::Colon)?;

            let body = if matches!(self.peek(), Token::LBrace) {
                let stmts = self.parse_block()?;
                MatchBody::Block(stmts)
            } else {
                let expr = self.parse_expr()?;
                MatchBody::Expr(expr)
            };

            arms.push(MatchArm { pattern, body });
            self.skip_newlines();
        }
        Ok(arms)
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        match self.peek().clone() {
            Token::Ident(ref s) if s == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            Token::Int(n) => {
                self.advance();
                Ok(Pattern::Int(n))
            }
            Token::Float(n) => {
                self.advance();
                Ok(Pattern::Float(n))
            }
            Token::Str(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Pattern::Str(s))
            }
            Token::RawStr(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Pattern::Str(s))
            }
            Token::True => {
                self.advance();
                Ok(Pattern::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Pattern::Bool(false))
            }
            Token::Nil => {
                self.advance();
                Ok(Pattern::Nil)
            }
            Token::Ident(ref s) => {
                let s = s.clone();
                // Check if this is a guard pattern (expression pattern)
                // An ident followed by something that looks like comparison, etc.
                // Peek at next token — if it's `:` then this is a simple ident match.
                // Otherwise it's a guard expression.
                if matches!(self.peek_at(1), Token::Colon) {
                    self.advance();
                    Ok(Pattern::Ident(s))
                } else {
                    // Guard expression: parse the whole expression
                    let expr = self.parse_expr()?;
                    Ok(Pattern::Guard(expr))
                }
            }
            Token::LParen => {
                // Tuple pattern or guard expression in parens
                // For now, treat as guard expression
                let expr = self.parse_expr()?;
                Ok(Pattern::Guard(expr))
            }
            Token::Minus => {
                // Negative number pattern
                self.advance();
                match self.advance() {
                    Token::Int(n) => Ok(Pattern::Int(-n)),
                    Token::Float(n) => Ok(Pattern::Float(-n)),
                    other => Err(self.error(format!(
                        "expected number after '-' in pattern, found {:?}",
                        other
                    ))),
                }
            }
            _ => {
                // Try to parse as guard expression
                let expr = self.parse_expr()?;
                Ok(Pattern::Guard(expr))
            }
        }
    }

    /// Parse select expression: `sel{ arms }`
    fn parse_select(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::Sel)?;
        self.expect(&Token::LBrace)?;

        let mut arms = Vec::new();
        self.skip_newlines();

        while !matches!(self.peek(), Token::RBrace | Token::Eof) {
            let arm = self.parse_select_arm()?;
            arms.push(arm);
            self.skip_newlines();
        }

        self.expect(&Token::RBrace)?;
        Ok(Expr::Select(arms))
    }

    /// Parse a single select arm:
    /// - Recv: `var = <- chan : { body }`
    /// - Send: `chan <- val : { body }`
    /// - Default: `_ : { body }`
    fn parse_select_arm(&mut self) -> Result<SelectArm, ParseError> {
        // Check for default arm: `_`
        if let Token::Ident(ref s) = self.peek().clone() {
            if s == "_" {
                self.advance(); // consume _
                self.expect(&Token::Colon)?;
                let body = self.parse_block()?;
                return Ok(SelectArm::Default(body));
            }
        }

        // Check for recv: `var = <- chan : { body }`
        // or send: `expr <- val : { body }`
        // Look ahead to distinguish:
        // recv: ident = <- expr : { }
        // send: expr <- expr : { }

        if matches!(self.peek(), Token::Ident(_)) && matches!(self.peek_at(1), Token::Eq) {
            // Recv arm: var = <- chan
            let var = match self.advance() {
                Token::Ident(s) => s,
                _ => unreachable!(),
            };
            self.expect(&Token::Eq)?;
            self.expect(&Token::ArrowLeft)?;
            let chan = self.parse_expr()?;
            self.expect(&Token::Colon)?;
            let body = self.parse_block()?;
            Ok(SelectArm::Recv { var, chan, body })
        } else {
            // Send arm: chan <- val : { body }
            let chan = self.parse_expr()?;

            // After the chan expression, we should now see `:` because
            // `chan <- val` was parsed as part of the expression (Expr::Send).
            // Actually, no. Let's reconsider.
            // In the select arm, `chan <- val` should be parsed as send.
            // But parse_expr -> parse_postfix would capture the `<-` and produce Send.
            // The chan expression includes the send. So actually, the expression
            // we parsed IS the send expression. We need to decompose it.

            self.expect(&Token::Colon)?;
            let body = self.parse_block()?;

            // Decompose the expression
            match chan {
                Expr::Send {
                    chan: ch,
                    value: val,
                } => Ok(SelectArm::Send {
                    chan: *ch,
                    value: *val,
                    body,
                }),
                _ => Err(self.error("expected send expression (chan <- val) in select arm".into())),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tok_lexer::lex;

    fn parse_str(input: &str) -> Result<Program, ParseError> {
        let tokens = lex(input).expect("lexer error");
        parse(tokens)
    }

    #[test]
    fn test_simple_expr() {
        let prog = parse_str("42").unwrap();
        assert_eq!(prog.len(), 1);
        assert!(matches!(prog[0], Stmt::Expr(Expr::Int(42))));
    }

    #[test]
    fn test_assignment() {
        let prog = parse_str("x=5").unwrap();
        assert_eq!(prog.len(), 1);
        match &prog[0] {
            Stmt::Assign { name, ty, value } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
                assert!(matches!(value, Expr::Int(5)));
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn test_func_decl() {
        let prog = parse_str("f add(a b)=a+b").unwrap();
        assert_eq!(prog.len(), 1);
        match &prog[0] {
            Stmt::FuncDecl { name, params, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected FuncDecl"),
        }
    }

    #[test]
    fn test_ternary() {
        let prog = parse_str("T?1:2").unwrap();
        assert_eq!(prog.len(), 1);
        match &prog[0] {
            Stmt::Expr(Expr::Ternary { .. }) => {}
            _ => panic!("expected Ternary"),
        }
    }

    #[test]
    fn test_array() {
        let prog = parse_str("[1 2 3]").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Array(elems)) => {
                assert_eq!(elems.len(), 3);
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_map() {
        let prog = parse_str("{a:1 b:2}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Map(entries)) => {
                assert_eq!(entries.len(), 2);
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn test_lambda() {
        let prog = parse_str(r"\(x)=x*2").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Lambda { params, .. }) => {
                assert_eq!(params.len(), 1);
            }
            _ => panic!("expected Lambda"),
        }
    }

    #[test]
    fn test_loop() {
        let prog = parse_str("~(i:0..5){p(i)}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Loop { clause, .. }) => match clause.as_ref() {
                LoopClause::ForRange { var, .. } => assert_eq!(var, "i"),
                _ => panic!("expected ForRange"),
            },
            _ => panic!("expected Loop with ForRange"),
        }
    }

    #[test]
    fn test_match() {
        let prog = parse_str("x?={1:\"one\";2:\"two\";_:\"other\"}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Match {
                subject: Some(_),
                arms,
            }) => {
                assert_eq!(arms.len(), 3);
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn test_pipe() {
        let prog = parse_str("5|>double").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Pipeline { .. }) => {}
            _ => panic!("expected Pipeline"),
        }
    }

    #[test]
    fn test_compound_assign() {
        let prog = parse_str("x+=1").unwrap();
        match &prog[0] {
            Stmt::CompoundAssign { name, op, .. } => {
                assert_eq!(name, "x");
                assert_eq!(*op, BinOp::Add);
            }
            _ => panic!("expected CompoundAssign"),
        }
    }

    #[test]
    fn test_member_assign() {
        let prog = parse_str("m.a=1").unwrap();
        match &prog[0] {
            Stmt::MemberAssign { field, .. } => {
                assert_eq!(field, "a");
            }
            _ => panic!("expected MemberAssign"),
        }
    }

    #[test]
    fn test_index_assign() {
        let prog = parse_str("a[0]=1").unwrap();
        match &prog[0] {
            Stmt::IndexAssign { .. } => {}
            _ => panic!("expected IndexAssign"),
        }
    }

    #[test]
    fn test_tuple_destructure() {
        let prog = parse_str("a b=(1 2)").unwrap();
        match &prog[0] {
            Stmt::TupleDestructure { names, .. } => {
                assert_eq!(names, &["a".to_string(), "b".to_string()]);
            }
            _ => panic!("expected TupleDestructure"),
        }
    }

    #[test]
    fn test_map_destructure() {
        let prog = parse_str("{x y}=point").unwrap();
        match &prog[0] {
            Stmt::MapDestructure { names, .. } => {
                assert_eq!(names, &["x".to_string(), "y".to_string()]);
            }
            _ => panic!("expected MapDestructure"),
        }
    }

    #[test]
    fn test_nil_coalesce() {
        let prog = parse_str("x??42").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::NilCoalesce { .. }) => {}
            _ => panic!("expected NilCoalesce"),
        }
    }

    #[test]
    fn test_optional_chain() {
        let prog = parse_str("a.?b").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::OptionalChain { field, .. }) => {
                assert_eq!(field, "b");
            }
            _ => panic!("expected OptionalChain"),
        }
    }

    #[test]
    fn test_error_propagate() {
        let prog = parse_str("x=div(10 2)?^").unwrap();
        // This should be an assignment where value is ErrorPropagate(Call(...))
        match &prog[0] {
            Stmt::Assign { value, .. } => {
                assert!(matches!(value, Expr::ErrorPropagate(_)));
            }
            _ => panic!("expected assignment with ErrorPropagate"),
        }
    }

    #[test]
    fn test_go() {
        let prog = parse_str("go{1+2}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Go(_)) => {}
            _ => panic!("expected Go"),
        }
    }

    #[test]
    fn test_receive() {
        let prog = parse_str("<-ch").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Receive(_)) => {}
            _ => panic!("expected Receive"),
        }
    }

    #[test]
    fn test_bare_import() {
        let prog = parse_str("@\"math\"").unwrap();
        match &prog[0] {
            Stmt::Import(path) => {
                assert_eq!(path, "math");
            }
            _ => panic!("expected Import stmt"),
        }
    }

    #[test]
    fn test_namespace_import() {
        let prog = parse_str("m=@\"math\"").unwrap();
        match &prog[0] {
            Stmt::Assign {
                value: Expr::Import(path),
                ..
            } => {
                assert_eq!(path, "math");
            }
            _ => panic!("expected Assign with Import"),
        }
    }

    #[test]
    fn test_nested_arrays() {
        // [[1 2] [3 4]] must NOT be parsed as [1 2][3 4]
        let prog = parse_str("[[1 2] [3 4]]").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Array(elems)) => {
                assert_eq!(elems.len(), 2);
                assert!(matches!(elems[0], Expr::Array(_)));
                assert!(matches!(elems[1], Expr::Array(_)));
            }
            _ => panic!("expected Array of Arrays"),
        }
    }

    #[test]
    fn test_tuples_in_array() {
        // [(1 N) (2 N)] must NOT be parsed as call
        let prog = parse_str("[(1 N) (2 N)]").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Array(elems)) => {
                assert_eq!(elems.len(), 2);
                assert!(matches!(elems[0], Expr::Tuple(_)));
                assert!(matches!(elems[1], Expr::Tuple(_)));
            }
            _ => panic!("expected Array of Tuples"),
        }
    }

    #[test]
    fn test_spread_in_array() {
        let prog = parse_str("[..x ..y]").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Array(elems)) => {
                assert_eq!(elems.len(), 2);
                assert!(matches!(elems[0], Expr::Spread(_)));
                assert!(matches!(elems[1], Expr::Spread(_)));
            }
            _ => panic!("expected Array with Spreads"),
        }
    }

    #[test]
    fn test_filter_reduce_chain() {
        // From test file: result=[1 2 3 4 5]?>(\(x)=x%2==0)/>\(a x)=a+x
        // The filter pred is wrapped in parens, then the reduce lambda is not.
        let prog = parse_str("a?>(\\(x)=x%2==0)/>\\(acc x)=acc+x").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Reduce { expr, .. }) => {
                assert!(matches!(expr.as_ref(), Expr::Filter { .. }));
            }
            _ => panic!("expected Reduce containing Filter"),
        }
    }

    #[test]
    fn test_conditional_return() {
        let prog = parse_str("x>0?^x").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::ConditionalReturn { .. }) => {}
            _ => panic!("expected ConditionalReturn"),
        }
    }

    #[test]
    fn test_break_continue() {
        let prog = parse_str("!").unwrap();
        assert!(matches!(prog[0], Stmt::Break));

        let prog = parse_str(">!").unwrap();
        assert!(matches!(prog[0], Stmt::Continue));
    }

    #[test]
    fn test_string_interp() {
        let prog = parse_str("\"hello {name}!\"").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Interp(parts)) => {
                assert_eq!(parts.len(), 3);
            }
            _ => panic!("expected Interp"),
        }
    }

    #[test]
    fn test_empty_map() {
        let prog = parse_str("{}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Map(entries)) => {
                assert!(entries.is_empty());
            }
            _ => panic!("expected empty Map"),
        }
    }

    #[test]
    fn test_select() {
        let prog = parse_str("sel{v=<-c:{v};_:{N}}").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Select(arms)) => {
                assert_eq!(arms.len(), 2);
            }
            _ => panic!("expected Select"),
        }
    }

    #[test]
    fn test_chained_tuple_access() {
        // t.1.1 should work
        let prog = parse_str("t.1.1").unwrap();
        match &prog[0] {
            Stmt::Expr(Expr::Member { expr, field }) => {
                assert_eq!(field, "1");
                match expr.as_ref() {
                    Expr::Member {
                        field: inner_field, ..
                    } => {
                        assert_eq!(inner_field, "1");
                    }
                    _ => panic!("expected nested Member"),
                }
            }
            _ => panic!("expected Member"),
        }
    }
}
