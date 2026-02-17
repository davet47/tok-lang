//! AST node types for the Tok language.

/// A program is a sequence of statements.
pub type Program = Vec<Stmt>;

/// Type expression for static type annotations.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Primitive: i, f, s, b, N, a
    Prim(PrimType),
    /// Array type: [T]
    Array(Box<TypeExpr>),
    /// Map type: {T} (string keys)
    Map(Box<TypeExpr>),
    /// Tuple type: (T U V)
    Tuple(Vec<TypeExpr>),
    /// Function type: \(T U):R
    Func(Vec<TypeExpr>, Box<TypeExpr>),
    /// Optional type: T?
    Optional(Box<TypeExpr>),
    /// Result type: R(T)
    Result(Box<TypeExpr>),
    /// Channel type: C(T)
    Channel(Box<TypeExpr>),
    /// Handle type: H(T)
    Handle(Box<TypeExpr>),
    /// Type variable: A, B, D, E, ...
    Var(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrimType {
    Int,   // i
    Float, // f
    Str,   // s
    Bool,  // b
    Nil,   // N
    Any,   // a
}

/// Statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Expression statement
    Expr(Expr),
    /// Variable assignment: name = expr (with optional type annotation)
    Assign {
        name: String,
        ty: Option<TypeExpr>,
        value: Expr,
    },
    /// Index assignment: expr[index] = value
    IndexAssign {
        target: Expr,
        index: Expr,
        value: Expr,
    },
    /// Member assignment: expr.field = value
    MemberAssign {
        target: Expr,
        field: String,
        value: Expr,
    },
    /// Compound assignment: name op= expr
    CompoundAssign {
        name: String,
        op: BinOp,
        value: Expr,
    },
    /// Compound index assignment: expr[index] op= value
    CompoundIndexAssign {
        target: Expr,
        index: Expr,
        op: BinOp,
        value: Expr,
    },
    /// Compound member assignment: expr.field op= value
    CompoundMemberAssign {
        target: Expr,
        field: String,
        op: BinOp,
        value: Expr,
    },
    /// Named function declaration: f name(params):rettype = expr | { body }
    FuncDecl {
        name: String,
        params: Vec<Param>,
        ret_type: Option<TypeExpr>,
        body: FuncBody,
    },
    /// Tuple destructuring: a b c = expr
    TupleDestructure { names: Vec<String>, value: Expr },
    /// Map destructuring: {a b c} = expr
    MapDestructure { names: Vec<String>, value: Expr },
    /// Array destructuring: [h ..t] = expr
    ArrayDestructure {
        head: String,
        tail: String,
        value: Expr,
    },
    /// Bare import: @"path"
    Import(String),
    /// Early return: ^expr or bare ^
    Return(Option<Expr>),
    /// Break: !
    Break,
    /// Continue: >!
    Continue,
}

/// Function/lambda parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub default: Option<Expr>,
    pub variadic: bool, // ..param
}

/// Function body — either single expression or block.
#[derive(Debug, Clone, PartialEq)]
pub enum FuncBody {
    Expr(Box<Expr>),
    Block(Vec<Stmt>),
}

/// Expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Nil,
    /// String interpolation: segments of literal strings and expressions
    Interp(Vec<InterpPart>),

    // Identifiers
    Ident(String),

    // Compound literals
    Array(Vec<Expr>),
    Map(Vec<(MapKey, Expr)>),
    Tuple(Vec<Expr>),

    // Range
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        inclusive: bool,
    },

    // Operators
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryOp {
        op: UnaryOp,
        expr: Box<Expr>,
    },

    // Access
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
    Member {
        expr: Box<Expr>,
        field: String,
    },
    OptionalChain {
        expr: Box<Expr>,
        field: String,
    },

    // Function call
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // Lambda
    Lambda {
        params: Vec<Param>,
        ret_type: Option<TypeExpr>,
        body: FuncBody,
    },

    // Control flow
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Option<Box<Expr>>,
    },
    Match {
        subject: Option<Box<Expr>>,
        arms: Vec<MatchArm>,
    },
    Loop {
        clause: Box<LoopClause>,
        body: Box<LoopBody>,
    },
    Block(Vec<Stmt>),

    // Pipeline operators
    Pipeline {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Filter {
        expr: Box<Expr>,
        pred: Box<Expr>,
    },
    Reduce {
        expr: Box<Expr>,
        init: Option<Box<Expr>>,
        func: Box<Expr>,
    },

    // Spread
    Spread(Box<Expr>),

    // Length
    Length(Box<Expr>),

    // Nil handling
    NilCoalesce {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Error propagation: expr?^ (postfix, no value after)
    ErrorPropagate(Box<Expr>),
    /// Conditional return: cond?^expr
    ConditionalReturn {
        cond: Box<Expr>,
        value: Box<Expr>,
    },

    // Concurrency
    Go(Box<Expr>),
    Receive(Box<Expr>),
    Send {
        chan: Box<Expr>,
        value: Box<Expr>,
    },
    Select(Vec<SelectArm>),

    // Modules
    Import(String),

    // Early return (as expression for `^expr` in expression context)
    Return(Option<Box<Expr>>),
    // Break/continue as expressions
    Break,
    Continue,
}

/// Map key — either a bare identifier (auto-stringified) or a string literal.
#[derive(Debug, Clone, PartialEq)]
pub enum MapKey {
    Ident(String),
    Str(String),
}

/// Interpolation part — either literal text or an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpPart {
    Lit(String),
    Expr(Expr),
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Append,
    Shr,
}

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// Match arm.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: MatchBody,
}

/// Match body — expression or block.
#[derive(Debug, Clone, PartialEq)]
pub enum MatchBody {
    Expr(Expr),
    Block(Vec<Stmt>),
}

/// Pattern in a match arm.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Nil,
    Ident(String),
    Tuple(Vec<Pattern>),
    /// Guard expression (condition-based match)
    Guard(Expr),
}

/// Loop clause.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopClause {
    /// While: ~(cond){body}
    While(Expr),
    /// Range: ~(i:0..10){body}
    ForRange { var: String, range: Expr },
    /// Foreach: ~(item:arr){body}
    ForEach { var: String, iter: Expr },
    /// Foreach with index/key: ~(i v:arr){body} or ~(k v:map){body}
    ForEachIndexed {
        idx_var: String,
        val_var: String,
        iter: Expr,
    },
    /// Infinite: ~{body}
    Infinite,
}

/// Loop body — block or collection expression.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopBody {
    Block(Vec<Stmt>),
    Collect(Expr), // ~(i:0..10)=expr
}

/// Select arm.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectArm {
    /// Receive: var = <-expr : {body}
    Recv {
        var: String,
        chan: Expr,
        body: Vec<Stmt>,
    },
    /// Send: expr <- expr : {body}
    Send {
        chan: Expr,
        value: Expr,
        body: Vec<Stmt>,
    },
    /// Default: _ : {body}
    Default(Vec<Stmt>),
}
