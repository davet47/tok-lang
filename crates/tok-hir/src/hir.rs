/// HIR (High-level Intermediate Representation) node types for the Tok language.
///
/// The HIR is a desugared version of the AST where high-level constructs
/// (compound assignments, string interpolation, pipelines, destructuring, etc.)
/// have been lowered into simpler primitives that are easy for codegen to consume.
///
/// Every `HirExpr` carries its resolved `Type` from the type checker.
use tok_types::Type;

/// HIR Program = sequence of HIR statements.
pub type HirProgram = Vec<HirStmt>;

/// HIR Statement -- simplified, no sugar.
#[derive(Debug, Clone)]
pub enum HirStmt {
    /// Variable assignment: name = expr
    Assign {
        name: String,
        ty: Type,
        value: HirExpr,
    },
    /// Function declaration
    FuncDecl {
        name: String,
        params: Vec<HirParam>,
        ret_type: Type,
        body: Vec<HirStmt>,
    },
    /// Index assignment: target[index] = value
    IndexAssign {
        target: HirExpr,
        index: HirExpr,
        value: HirExpr,
    },
    /// Member assignment: target.field = value
    MemberAssign {
        target: HirExpr,
        field: String,
        value: HirExpr,
    },
    /// Expression statement (for side effects)
    Expr(HirExpr),
    /// Return
    Return(Option<HirExpr>),
    /// Break
    Break,
    /// Continue
    Continue,
    /// Import (bare: merge exports into scope)
    Import(String),
}

/// Function/lambda parameter.
#[derive(Debug, Clone)]
pub struct HirParam {
    pub name: String,
    pub ty: Type,
}

/// HIR Expression -- every node carries its Type.
#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
}

impl HirExpr {
    /// Convenience constructor.
    pub fn new(kind: HirExprKind, ty: Type) -> Self {
        HirExpr { kind, ty }
    }
}

/// HIR Expression kinds -- desugared, no compound forms.
#[derive(Debug, Clone)]
pub enum HirExprKind {
    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Nil,

    // Identifiers
    Ident(String),

    // Compound literals
    Array(Vec<HirExpr>),
    Map(Vec<(String, HirExpr)>),
    Tuple(Vec<HirExpr>),

    // Operators
    BinOp {
        op: HirBinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
    UnaryOp {
        op: HirUnaryOp,
        operand: Box<HirExpr>,
    },

    // Access
    Index {
        target: Box<HirExpr>,
        index: Box<HirExpr>,
    },
    Member {
        target: Box<HirExpr>,
        field: String,
    },

    // Function call
    Call {
        func: Box<HirExpr>,
        args: Vec<HirExpr>,
    },
    /// Runtime call -- calls an extern "C" runtime function by name.
    RuntimeCall {
        name: String,
        args: Vec<HirExpr>,
    },

    // Lambda / closure
    Lambda {
        params: Vec<HirParam>,
        ret_type: Type,
        body: Vec<HirStmt>,
    },

    // Control flow (simplified)
    If {
        cond: Box<HirExpr>,
        then_body: Vec<HirStmt>,
        then_expr: Option<Box<HirExpr>>,
        else_body: Vec<HirStmt>,
        else_expr: Option<Box<HirExpr>>,
    },

    // Loop (simplified -- always block body, no collect)
    Loop {
        kind: Box<HirLoopKind>,
        body: Vec<HirStmt>,
    },

    // Block
    Block {
        stmts: Vec<HirStmt>,
        expr: Option<Box<HirExpr>>,
    },

    // Length
    Length(Box<HirExpr>),

    // Range
    Range {
        start: Box<HirExpr>,
        end: Box<HirExpr>,
        inclusive: bool,
    },

    // Concurrency
    Go(Box<HirExpr>),
    Receive(Box<HirExpr>),
    Send {
        chan: Box<HirExpr>,
        value: Box<HirExpr>,
    },
    Select(Vec<HirSelectArm>),
}

/// Binary operators (same set as AST, but separate type for HIR independence).
#[derive(Debug, Clone, Copy)]
pub enum HirBinOp {
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
    Shr,
}

/// Unary operators.
#[derive(Debug, Clone, Copy)]
pub enum HirUnaryOp {
    Neg,
    Not,
}

/// Loop kinds (simplified from AST -- ForRange unpacks Range expr).
#[derive(Debug, Clone)]
pub enum HirLoopKind {
    While(Box<HirExpr>),
    ForRange {
        var: String,
        start: HirExpr,
        end: HirExpr,
        inclusive: bool,
    },
    ForEach {
        var: String,
        iter: HirExpr,
    },
    ForEachIndexed {
        idx_var: String,
        val_var: String,
        iter: HirExpr,
    },
    Infinite,
}

/// Select arm (channel multiplexing).
#[derive(Debug, Clone)]
pub enum HirSelectArm {
    Recv {
        var: String,
        chan: HirExpr,
        body: Vec<HirStmt>,
    },
    Send {
        chan: HirExpr,
        value: HirExpr,
        body: Vec<HirStmt>,
    },
    Default(Vec<HirStmt>),
}
