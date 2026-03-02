use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_program(&mut self, program: &Program) -> HirProgram {
        let mut out = Vec::new();
        for stmt in program {
            self.lower_stmt(stmt, &mut out);
        }
        out
    }

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt, out: &mut Vec<HirStmt>) {
        match stmt {
            Stmt::Expr(expr) => {
                let hir_expr = self.lower_expr(expr);
                out.push(HirStmt::Expr(hir_expr));
            }

            Stmt::Assign { name, value, .. } => {
                let hir_value = self.lower_expr(value);
                let ty = hir_value.ty.clone();
                // Track the local variable type for subsequent references
                self.define_local(name, ty.clone());

                // Track method_fields: if the value was a Map or ProtoInit with methods,
                // associate the variable name with those method field names.
                if let Some(pending) = self.method_fields.remove("__pending_map__") {
                    self.method_fields.insert(name.clone(), pending);
                }
                // If ProtoInit lowering stored methods under its tmp var, transfer to this name
                if let Expr::ProtoInit { proto, .. } = value {
                    // The lower_proto_init already computed and stored under tmp;
                    // we need to find the gensym'd tmp name from the block.
                    // Instead, check if the lowered value is a Block and extract the tmp name.
                    if let HirExprKind::Block {
                        expr: Some(ref result),
                        ..
                    } = hir_value.kind
                    {
                        if let HirExprKind::Ident(ref tmp_name) = result.kind {
                            if let Some(methods) = self.method_fields.remove(tmp_name) {
                                self.method_fields.insert(name.clone(), methods);
                            }
                        }
                    }
                    // Also inherit from proto if not already handled
                    if !self.method_fields.contains_key(name) {
                        if let Expr::Ident(proto_name) = proto.as_ref() {
                            if let Some(methods) = self.method_fields.get(proto_name).cloned() {
                                self.method_fields.insert(name.clone(), methods);
                            }
                        }
                    }
                }
                // If assigning from another variable, propagate method_fields
                if let Expr::Ident(src_name) = value {
                    if let Some(methods) = self.method_fields.get(src_name).cloned() {
                        self.method_fields.insert(name.clone(), methods);
                    }
                }

                out.push(HirStmt::Assign {
                    name: name.clone(),
                    ty,
                    value: hir_value,
                });
            }

            Stmt::FuncDecl {
                name,
                params,
                ret_type: _,
                body,
            } => {
                let hir_params = self.lower_params(params);
                let ret_type = self.func_ret_type(name);
                // Push scope for function body with param types
                self.push_scope();
                for hp in &hir_params {
                    self.define_local(&hp.name, hp.ty.clone());
                }
                let hir_body = self.lower_func_body(body);
                self.pop_scope();
                out.push(HirStmt::FuncDecl {
                    name: name.clone(),
                    params: hir_params,
                    ret_type,
                    body: hir_body,
                });
            }

            Stmt::IndexAssign {
                target,
                index,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let hir_value = self.lower_expr(value);
                out.push(HirStmt::IndexAssign {
                    target: hir_target,
                    index: hir_index,
                    value: hir_value,
                });
            }

            Stmt::MemberAssign {
                target,
                field,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_value = self.lower_expr(value);
                out.push(HirStmt::MemberAssign {
                    target: hir_target,
                    field: field.clone(),
                    value: hir_value,
                });
            }

            // Desugar: x += 1 -> x = x + 1  (or x <<= v -> x = push(x, v))
            Stmt::CompoundAssign { name, op, value } => {
                let hir_value = self.lower_expr(value);
                let var_ty = self.var_type(name);
                let result_ty = self.infer_binop_type(op, &var_ty, &hir_value.ty);
                let ident = HirExpr::new(HirExprKind::Ident(name.clone()), var_ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![ident, hir_value],
                        },
                        result_ty.clone(),
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(ident),
                            right: Box::new(hir_value),
                        },
                        result_ty.clone(),
                    )
                };
                out.push(HirStmt::Assign {
                    name: name.clone(),
                    ty: result_ty,
                    value: rhs,
                });
            }

            // Desugar: arr[i] += 1 -> arr[i] = arr[i] + 1
            Stmt::CompoundIndexAssign {
                target,
                index,
                op,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let hir_value = self.lower_expr(value);

                let current = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(hir_target.clone()),
                        index: Box::new(hir_index.clone()),
                    },
                    Type::Any,
                );
                let result_ty = self.infer_binop_type(op, &current.ty, &hir_value.ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![current, hir_value],
                        },
                        result_ty,
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(current),
                            right: Box::new(hir_value),
                        },
                        result_ty,
                    )
                };
                out.push(HirStmt::IndexAssign {
                    target: hir_target,
                    index: hir_index,
                    value: rhs,
                });
            }

            // Desugar: m.x += 1 -> m.x = m.x + 1
            Stmt::CompoundMemberAssign {
                target,
                field,
                op,
                value,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_value = self.lower_expr(value);

                let current = HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(hir_target.clone()),
                        field: field.clone(),
                    },
                    Type::Any,
                );
                let result_ty = self.infer_binop_type(op, &current.ty, &hir_value.ty);
                let rhs = if matches!(op, BinOp::Append) {
                    HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![current, hir_value],
                        },
                        result_ty,
                    )
                } else {
                    HirExpr::new(
                        HirExprKind::BinOp {
                            op: self.lower_binop(op),
                            left: Box::new(current),
                            right: Box::new(hir_value),
                        },
                        result_ty,
                    )
                };
                out.push(HirStmt::MemberAssign {
                    target: hir_target,
                    field: field.clone(),
                    value: rhs,
                });
            }

            // Desugar: a b = expr -> _tmp = expr; a = _tmp.0; b = _tmp.1
            Stmt::TupleDestructure { names, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                for (i, name) in names.iter().enumerate() {
                    let elem_ty = match &val_ty {
                        Type::Tuple(elts) => elts.get(i).cloned().unwrap_or(Type::Any),
                        _ => Type::Any,
                    };
                    self.define_local(name, elem_ty.clone());
                    let index_expr = HirExpr::new(
                        HirExprKind::Index {
                            target: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                val_ty.clone(),
                            )),
                            index: Box::new(HirExpr::new(HirExprKind::Int(i as i64), Type::Int)),
                        },
                        elem_ty.clone(),
                    );
                    out.push(HirStmt::Assign {
                        name: name.clone(),
                        ty: elem_ty,
                        value: index_expr,
                    });
                }
            }

            // Desugar: {a b} = expr -> _tmp = expr; a = _tmp.a; b = _tmp.b
            Stmt::MapDestructure { names, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                let elem_ty = match &val_ty {
                    Type::Map(inner) => *inner.clone(),
                    _ => Type::Any,
                };

                for name in names {
                    self.define_local(name, elem_ty.clone());
                    let member_expr = HirExpr::new(
                        HirExprKind::Member {
                            target: Box::new(HirExpr::new(
                                HirExprKind::Ident(tmp.clone()),
                                val_ty.clone(),
                            )),
                            field: name.clone(),
                        },
                        elem_ty.clone(),
                    );
                    out.push(HirStmt::Assign {
                        name: name.clone(),
                        ty: elem_ty.clone(),
                        value: member_expr,
                    });
                }
            }

            // Desugar: [h ..t] = expr -> _tmp = expr; h = _tmp[0]; t = slice(_tmp, 1, len(_tmp))
            Stmt::ArrayDestructure { head, tail, value } => {
                let hir_value = self.lower_expr(value);
                let val_ty = hir_value.ty.clone();
                let tmp = self.gensym();

                out.push(HirStmt::Assign {
                    name: tmp.clone(),
                    ty: val_ty.clone(),
                    value: hir_value,
                });

                let elem_ty = match &val_ty {
                    Type::Array(inner) => *inner.clone(),
                    _ => Type::Any,
                };

                // h = _tmp[0]
                self.define_local(head, elem_ty.clone());
                let head_expr = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            val_ty.clone(),
                        )),
                        index: Box::new(HirExpr::new(HirExprKind::Int(0), Type::Int)),
                    },
                    elem_ty.clone(),
                );
                out.push(HirStmt::Assign {
                    name: head.clone(),
                    ty: elem_ty,
                    value: head_expr,
                });

                // t = tok_array_slice(_tmp, 1, #_tmp)
                let tmp_ident = HirExpr::new(HirExprKind::Ident(tmp.clone()), val_ty.clone());
                let tail_ty = val_ty.clone();
                let tail_expr = HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_slice".to_string(),
                        args: vec![
                            tmp_ident.clone(),
                            HirExpr::new(HirExprKind::Int(1), Type::Int),
                            HirExpr::new(HirExprKind::Length(Box::new(tmp_ident)), Type::Int),
                        ],
                    },
                    val_ty,
                );
                self.define_local(tail, tail_ty);
                out.push(HirStmt::Assign {
                    name: tail.clone(),
                    ty: tail_expr.ty.clone(),
                    value: tail_expr,
                });
            }

            Stmt::Import(path) => {
                out.push(HirStmt::Import(path.clone()));
            }

            Stmt::Return(expr) => {
                let hir_expr = expr.as_ref().map(|e| self.lower_expr(e));
                out.push(HirStmt::Return(hir_expr));
            }

            Stmt::Break => {
                out.push(HirStmt::Break);
            }

            Stmt::Continue => {
                out.push(HirStmt::Continue);
            }
        }
    }
}
