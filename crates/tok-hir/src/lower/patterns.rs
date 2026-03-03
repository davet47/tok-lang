use super::*;

impl<'a> Lowerer<'a> {
    /// Lower match to if-else chain.
    pub(super) fn lower_match(
        &mut self,
        subject: &Option<Box<Expr>>,
        arms: &[ast::MatchArm],
    ) -> HirExpr {
        let hir_subject = subject.as_ref().map(|s| self.lower_expr(s));
        let (tmp_name, tmp_stmts) = if let Some(ref subj) = hir_subject {
            let tmp = self.gensym();
            let stmt = HirStmt::Assign {
                name: tmp.clone(),
                ty: subj.ty.clone(),
                value: subj.clone(),
            };
            (Some(tmp), vec![stmt])
        } else {
            (None, vec![])
        };

        let subject_ty = hir_subject
            .as_ref()
            .map(|s| s.ty.clone())
            .unwrap_or(Type::Any);

        let if_chain = self.lower_match_arms(arms, &tmp_name, &subject_ty);

        if tmp_stmts.is_empty() {
            if_chain
        } else {
            let result_ty = if_chain.ty.clone();
            HirExpr::new(
                HirExprKind::Block {
                    stmts: tmp_stmts,
                    expr: Some(Box::new(if_chain)),
                },
                result_ty,
            )
        }
    }

    fn lower_match_arms(
        &mut self,
        arms: &[ast::MatchArm],
        subject_tmp: &Option<String>,
        subject_ty: &Type,
    ) -> HirExpr {
        if arms.is_empty() {
            return HirExpr::new(HirExprKind::Nil, Type::Nil);
        }

        let arm = &arms[0];
        let rest = &arms[1..];

        let body_expr = self.lower_match_body(&arm.body);
        let body_ty = body_expr.ty.clone();

        match &arm.pattern {
            Pattern::Wildcard => {
                // Wildcard matches everything -- this is the final else
                body_expr
            }
            Pattern::Guard(guard_expr) => {
                // Guard: condition is the guard expression itself
                let cond = self.lower_expr(guard_expr);
                let else_expr = self.lower_match_arms(rest, subject_tmp, subject_ty);
                let result_ty = tok_types::unify(&body_ty, &else_expr.ty);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(body_expr)),
                        else_body: vec![],
                        else_expr: Some(Box::new(else_expr)),
                    },
                    result_ty,
                )
            }
            _ => {
                // Value pattern: compare subject with pattern value
                let pat_expr = self.pattern_to_expr(&arm.pattern);
                let cond = if let Some(ref tmp) = subject_tmp {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: HirBinOp::Eq,
                            left: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                subject_ty.clone(),
                            )),
                            right: Box::new(pat_expr),
                        },
                        Type::Bool,
                    )
                } else {
                    // No subject -- pattern itself is the condition (guard-like)
                    pat_expr
                };
                let else_expr = self.lower_match_arms(rest, subject_tmp, subject_ty);
                let result_ty = tok_types::unify(&body_ty, &else_expr.ty);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(body_expr)),
                        else_body: vec![],
                        else_expr: Some(Box::new(else_expr)),
                    },
                    result_ty,
                )
            }
        }
    }

    fn pattern_to_expr(&self, pat: &Pattern) -> HirExpr {
        match pat {
            Pattern::Int(v) => HirExpr::new(HirExprKind::Int(*v), Type::Int),
            Pattern::Float(v) => HirExpr::new(HirExprKind::Float(*v), Type::Float),
            Pattern::Str(v) => HirExpr::new(HirExprKind::Str(v.clone()), Type::Str),
            Pattern::Bool(v) => HirExpr::new(HirExprKind::Bool(*v), Type::Bool),
            Pattern::Nil => HirExpr::new(HirExprKind::Nil, Type::Nil),
            Pattern::Ident(name) => {
                let ty = self.var_type(name);
                HirExpr::new(HirExprKind::Ident(name.clone()), ty)
            }
            Pattern::Wildcard => HirExpr::new(HirExprKind::Bool(true), Type::Bool),
            Pattern::Tuple(pats) => {
                let hir_elts: Vec<HirExpr> = pats.iter().map(|p| self.pattern_to_expr(p)).collect();
                let tys: Vec<Type> = hir_elts.iter().map(|e| e.ty.clone()).collect();
                HirExpr::new(HirExprKind::Tuple(hir_elts), Type::Tuple(tys))
            }
            Pattern::Guard(_expr) => {
                // Guards are handled in lower_match_arms before pattern_to_expr
                // is called. Reaching here indicates a logic error.
                eprintln!("warning: Pattern::Guard reached pattern_to_expr (should be handled in lower_match_arms)");
                HirExpr::new(HirExprKind::Bool(true), Type::Bool)
            }
        }
    }

    pub(super) fn lower_match_body(&mut self, body: &MatchBody) -> HirExpr {
        match body {
            MatchBody::Expr(e) => self.lower_expr(e),
            MatchBody::Block(stmts) => self.lower_block(stmts),
        }
    }
}
