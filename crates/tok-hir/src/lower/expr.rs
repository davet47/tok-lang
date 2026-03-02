use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_expr(&mut self, expr: &Expr) -> HirExpr {
        match expr {
            // Literals -- direct mapping
            Expr::Int(v) => HirExpr::new(HirExprKind::Int(*v), Type::Int),
            Expr::Float(v) => HirExpr::new(HirExprKind::Float(*v), Type::Float),
            Expr::Str(v) => HirExpr::new(HirExprKind::Str(v.clone()), Type::Str),
            Expr::Bool(v) => HirExpr::new(HirExprKind::Bool(*v), Type::Bool),
            Expr::Nil => HirExpr::new(HirExprKind::Nil, Type::Nil),

            // Identifiers
            Expr::Ident(name) => {
                let ty = self.var_type(name);
                HirExpr::new(HirExprKind::Ident(name.clone()), ty)
            }

            // Desugar string interpolation:
            // "hello {name}!" -> tok_string_concat(tok_string_concat("hello ", tok_value_to_string(name)), "!")
            Expr::Interp(parts) => self.lower_interp(parts),

            // Compound literals
            Expr::Array(elts) => self.lower_array(elts),

            Expr::Map(pairs) => {
                // Track which fields are methods (lambdas)
                let mut methods = std::collections::HashSet::new();
                let hir_pairs: Vec<(String, HirExpr)> = pairs
                    .iter()
                    .map(|(key, val)| {
                        let key_str = match key {
                            MapKey::Ident(s) | MapKey::Str(s) => s.clone(),
                        };
                        let is_method = matches!(val, Expr::Lambda { .. });
                        let hir_val = if is_method {
                            methods.insert(key_str.clone());
                            self.lower_method_lambda(val)
                        } else {
                            self.lower_expr(val)
                        };
                        (key_str, hir_val)
                    })
                    .collect();
                let val_ty = if hir_pairs.is_empty() {
                    Type::Any
                } else {
                    let mut t = hir_pairs[0].1.ty.clone();
                    for (_, v) in &hir_pairs[1..] {
                        t = tok_types::unify(&t, &v.ty);
                    }
                    t
                };
                // Store methods info temporarily using _map_methods_ prefix
                // The assign handler will pick this up
                if !methods.is_empty() {
                    self.method_fields
                        .insert("__pending_map__".to_string(), methods);
                }
                HirExpr::new(HirExprKind::Map(hir_pairs), Type::Map(Box::new(val_ty)))
            }

            Expr::Tuple(elts) => {
                let hir_elts: Vec<HirExpr> = elts.iter().map(|e| self.lower_expr(e)).collect();
                let tys: Vec<Type> = hir_elts.iter().map(|e| e.ty.clone()).collect();
                HirExpr::new(HirExprKind::Tuple(hir_elts), Type::Tuple(tys))
            }

            // Range
            Expr::Range {
                start,
                end,
                inclusive,
            } => {
                let hir_start = self.lower_expr(start);
                let hir_end = self.lower_expr(end);
                HirExpr::new(
                    HirExprKind::Range {
                        start: Box::new(hir_start),
                        end: Box::new(hir_end),
                        inclusive: *inclusive,
                    },
                    Type::Range,
                )
            }

            // Binary ops
            Expr::BinOp { op, left, right } => {
                if matches!(op, BinOp::Append) {
                    let hir_left = self.lower_expr(left);
                    let hir_right = self.lower_expr(right);
                    let ty = self.infer_binop_type(op, &hir_left.ty, &hir_right.ty);
                    return HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![hir_left, hir_right],
                        },
                        ty,
                    );
                }
                let hir_left = self.lower_expr(left);
                let hir_right = self.lower_expr(right);
                let ty = self.infer_binop_type(op, &hir_left.ty, &hir_right.ty);
                HirExpr::new(
                    HirExprKind::BinOp {
                        op: self.lower_binop(op),
                        left: Box::new(hir_left),
                        right: Box::new(hir_right),
                    },
                    ty,
                )
            }

            // Unary ops
            Expr::UnaryOp { op, expr } => {
                let hir_expr = self.lower_expr(expr);
                let ty = match op {
                    UnaryOp::Neg => match &hir_expr.ty {
                        Type::Int => Type::Int,
                        Type::Float => Type::Float,
                        _ => Type::Any,
                    },
                    UnaryOp::Not => Type::Bool,
                };
                HirExpr::new(
                    HirExprKind::UnaryOp {
                        op: self.lower_unaryop(op),
                        operand: Box::new(hir_expr),
                    },
                    ty,
                )
            }

            // Index
            Expr::Index {
                expr: target,
                index,
            } => {
                let hir_target = self.lower_expr(target);
                let hir_index = self.lower_expr(index);
                let ty = match &hir_target.ty {
                    Type::Array(inner) => *inner.clone(),
                    Type::Map(inner) => *inner.clone(),
                    Type::Tuple(elts) => {
                        if let HirExprKind::Int(i) = &hir_index.kind {
                            elts.get(*i as usize).cloned().unwrap_or(Type::Any)
                        } else {
                            Type::Any
                        }
                    }
                    Type::Str => Type::Str,
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(hir_target),
                        index: Box::new(hir_index),
                    },
                    ty,
                )
            }

            // Member
            Expr::Member {
                expr: target,
                field,
            } => {
                let hir_target = self.lower_expr(target);
                let ty = self.infer_member_type(&hir_target.ty, field);
                HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(hir_target),
                        field: field.clone(),
                    },
                    ty,
                )
            }

            // Desugar optional chain: expr.?field ->
            //   if expr != Nil { expr.field } else { Nil }
            Expr::OptionalChain {
                expr: target,
                field,
            } => {
                let hir_target = self.lower_expr(target);
                let target_ty = hir_target.ty.clone();
                let field_ty = self.infer_member_type(&target_ty, field);
                let result_ty = Type::Optional(Box::new(field_ty.clone()));

                let tmp = self.gensym();
                // Build: if _tmp != Nil then _tmp.field else Nil
                let cond = HirExpr::new(
                    HirExprKind::BinOp {
                        op: HirBinOp::Neq,
                        left: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            target_ty.clone(),
                        )),
                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                    },
                    Type::Bool,
                );
                let then_expr = HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(HirExpr::new(
                            HirExprKind::Ident(tmp.clone()),
                            target_ty.clone(),
                        )),
                        field: field.clone(),
                    },
                    field_ty,
                );

                // Wrap in a block: { _tmp = target; if _tmp != Nil ... }
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp,
                            ty: target_ty,
                            value: hir_target,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(cond),
                                then_body: vec![],
                                then_expr: Some(Box::new(then_expr)),
                                else_body: vec![],
                                else_expr: Some(Box::new(HirExpr::new(
                                    HirExprKind::Nil,
                                    Type::Nil,
                                ))),
                            },
                            result_ty.clone(),
                        ))),
                    },
                    result_ty,
                )
            }

            // Function call
            Expr::Call { func, args } => {
                // Method call self-injection: obj.method(args) → method(__self=obj, args)
                if let Expr::Member {
                    expr: ref target,
                    field,
                } = func.as_ref()
                {
                    if let Expr::Ident(ref obj_name) = target.as_ref() {
                        if self
                            .method_fields
                            .get(obj_name)
                            .is_some_and(|methods| methods.contains(field))
                        {
                            // It's a method call — inject obj as first arg
                            let hir_func = self.lower_expr(func);
                            let obj_hir = HirExpr::new(
                                HirExprKind::Ident(obj_name.clone()),
                                Type::Map(Box::new(Type::Any)),
                            );
                            let mut hir_args = vec![obj_hir];
                            hir_args.extend(args.iter().map(|a| self.lower_expr(a)));
                            let ret_ty = match &hir_func.ty {
                                Type::Func(ft) => *ft.ret.clone(),
                                _ => Type::Any,
                            };
                            return HirExpr::new(
                                HirExprKind::Call {
                                    func: Box::new(hir_func),
                                    args: hir_args,
                                },
                                ret_ty,
                            );
                        }
                    }
                }

                let has_spread = args.iter().any(|a| matches!(a, Expr::Spread(_)));
                if has_spread {
                    return self.lower_call_with_spread(func, args);
                }
                let hir_func = self.lower_expr(func);
                let mut hir_args: Vec<HirExpr> = args.iter().map(|a| self.lower_expr(a)).collect();
                // Pad missing args with default expressions for named functions
                if let Expr::Ident(name) = func.as_ref() {
                    let missing: Vec<Expr> = self
                        .func_defaults
                        .get(name.as_str())
                        .into_iter()
                        .flat_map(|defaults| {
                            defaults.iter().skip(args.len()).filter_map(|d| d.clone())
                        })
                        .collect();
                    for default_expr in &missing {
                        hir_args.push(self.lower_expr(default_expr));
                    }
                    // Wrap extra args into array for variadic functions
                    if let Some(&n_fixed) = self.func_variadic.get(name.as_str()) {
                        if hir_args.len() >= n_fixed {
                            let fixed = hir_args[..n_fixed].to_vec();
                            let variadic_elts = hir_args[n_fixed..].to_vec();
                            let arr_ty = Type::Array(Box::new(Type::Any));
                            let arr_expr = HirExpr::new(HirExprKind::Array(variadic_elts), arr_ty);
                            hir_args = fixed;
                            hir_args.push(arr_expr);
                        }
                    }
                }
                let ret_ty = match &hir_func.ty {
                    Type::Func(ft) => *ft.ret.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(
                    HirExprKind::Call {
                        func: Box::new(hir_func),
                        args: hir_args,
                    },
                    ret_ty,
                )
            }

            // Lambda
            Expr::Lambda {
                params,
                ret_type: _,
                body,
            } => {
                let hir_params = self.lower_params(params);
                self.push_scope();
                for hp in &hir_params {
                    self.define_local(&hp.name, hp.ty.clone());
                }
                let hir_body = self.lower_func_body(body);
                self.pop_scope();
                let param_types: Vec<tok_types::ParamType> = hir_params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: p.ty.clone(),
                        has_default: p.has_default,
                    })
                    .collect();
                let ret_ty = Type::Any; // simplified
                let is_variadic = params.last().is_some_and(|p| p.variadic);
                let func_ty = Type::Func(tok_types::FuncType {
                    params: param_types,
                    ret: Box::new(ret_ty.clone()),
                    variadic: is_variadic,
                });
                HirExpr::new(
                    HirExprKind::Lambda {
                        params: hir_params,
                        ret_type: ret_ty,
                        body: hir_body,
                    },
                    func_ty,
                )
            }

            // Ternary -> If
            Expr::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let hir_cond = self.lower_expr(cond);
                let hir_then = self.lower_expr(then_expr);
                let then_ty = hir_then.ty.clone();
                let (else_body, hir_else, result_ty) = if let Some(else_e) = else_expr {
                    let hir_else = self.lower_expr(else_e);
                    let ty = tok_types::unify(&then_ty, &hir_else.ty);
                    (vec![], Some(Box::new(hir_else)), ty)
                } else {
                    (vec![], None, then_ty)
                };
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(hir_cond),
                        then_body: vec![],
                        then_expr: Some(Box::new(hir_then)),
                        else_body,
                        else_expr: hir_else,
                    },
                    result_ty,
                )
            }

            // Desugar Match -> if-else chain
            Expr::Match { subject, arms } => self.lower_match(subject, arms),

            // Loop
            Expr::Loop { clause, body } => self.lower_loop(clause, body),

            // Block
            Expr::Block(stmts) => self.lower_block(stmts),

            // Desugar pipeline: x |> f -> f(x)
            //                   x |> f(y) -> f(x, y)
            Expr::Pipeline { left, right } => self.lower_pipeline(left, right),

            // Desugar filter: arr ?> pred -> RuntimeCall("tok_array_filter", [arr, pred])
            Expr::Filter { expr, pred } => {
                let hir_arr = self.lower_expr(expr);
                let arr_ty = hir_arr.ty.clone();
                let hir_pred = self.lower_expr(pred);
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_filter".to_string(),
                        args: vec![hir_arr, hir_pred],
                    },
                    arr_ty,
                )
            }

            // Desugar reduce: arr /> init fn -> RuntimeCall("tok_array_reduce", [arr, init, fn])
            Expr::Reduce { expr, init, func } => {
                let hir_arr = self.lower_expr(expr);
                let hir_func = self.lower_expr(func);
                let result_ty = if let Some(init_e) = init {
                    self.infer_expr_type(init_e)
                } else {
                    match &hir_arr.ty {
                        Type::Array(inner) => *inner.clone(),
                        _ => Type::Any,
                    }
                };
                let hir_init = if let Some(init_e) = init {
                    self.lower_expr(init_e)
                } else {
                    // When no init, pass Nil as sentinel (runtime handles first-element init)
                    HirExpr::new(HirExprKind::Nil, Type::Nil)
                };
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_array_reduce".to_string(),
                        args: vec![hir_arr, hir_init, hir_func],
                    },
                    result_ty,
                )
            }

            // Spread -- should only appear inside arrays, handled in lower_array
            Expr::Spread(inner) => {
                // If encountered outside an array context, just lower the inner expression
                self.lower_expr(inner)
            }

            // Length
            Expr::Length(inner) => {
                let hir_inner = self.lower_expr(inner);
                HirExpr::new(HirExprKind::Length(Box::new(hir_inner)), Type::Int)
            }

            // Desugar nil coalesce: left ?? right -> if left != Nil { left } else { right }
            Expr::NilCoalesce { left, right } => {
                let hir_left = self.lower_expr(left);
                let hir_right = self.lower_expr(right);
                let left_ty = hir_left.ty.clone();
                let result_ty = match &left_ty {
                    Type::Optional(inner) => tok_types::unify(inner, &hir_right.ty),
                    Type::Nil => hir_right.ty.clone(),
                    _ => tok_types::unify(&left_ty, &hir_right.ty),
                };

                let tmp = self.gensym();
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp.clone(),
                            ty: left_ty.clone(),
                            value: hir_left,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(HirExpr::new(
                                    HirExprKind::BinOp {
                                        op: HirBinOp::Neq,
                                        left: Box::new(HirExpr::new(
                                            HirExprKind::Ident(tmp.clone()),
                                            left_ty.clone(),
                                        )),
                                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                                    },
                                    Type::Bool,
                                )),
                                then_body: vec![],
                                then_expr: Some(Box::new(HirExpr::new(
                                    HirExprKind::Ident(tmp),
                                    left_ty,
                                ))),
                                else_body: vec![],
                                else_expr: Some(Box::new(hir_right)),
                            },
                            result_ty.clone(),
                        ))),
                    },
                    result_ty,
                )
            }

            // Desugar error propagation: expr?^ ->
            //   _tmp = expr; if _tmp.1 != Nil { return _tmp } else { _tmp.0 }
            Expr::ErrorPropagate(inner) => {
                let hir_inner = self.lower_expr(inner);
                let inner_ty = hir_inner.ty.clone();
                let ok_ty = match &inner_ty {
                    Type::Result(ok_ty) => *ok_ty.clone(),
                    Type::Tuple(elts) if elts.len() == 2 => elts[0].clone(),
                    _ => Type::Any,
                };

                let tmp = self.gensym();
                let tmp_ident = HirExpr::new(HirExprKind::Ident(tmp.clone()), inner_ty.clone());

                // _tmp.1 (error field)
                let err_field = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident.clone()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(1), Type::Int)),
                    },
                    Type::Any,
                );

                // _tmp.0 (ok field)
                let ok_field = HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident.clone()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(0), Type::Int)),
                    },
                    ok_ty.clone(),
                );

                // if _tmp.1 != Nil then return _tmp else _tmp.0
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Assign {
                            name: tmp,
                            ty: inner_ty,
                            value: hir_inner,
                        }],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::If {
                                cond: Box::new(HirExpr::new(
                                    HirExprKind::BinOp {
                                        op: HirBinOp::Neq,
                                        left: Box::new(err_field),
                                        right: Box::new(HirExpr::new(HirExprKind::Nil, Type::Nil)),
                                    },
                                    Type::Bool,
                                )),
                                then_body: vec![HirStmt::Return(Some(tmp_ident))],
                                then_expr: None,
                                else_body: vec![],
                                else_expr: Some(Box::new(ok_field)),
                            },
                            ok_ty.clone(),
                        ))),
                    },
                    ok_ty,
                )
            }

            // Desugar conditional return: cond?^value -> if cond { return value }
            Expr::ConditionalReturn { cond, value } => {
                let hir_cond = self.lower_expr(cond);
                let hir_value = self.lower_expr(value);
                HirExpr::new(
                    HirExprKind::If {
                        cond: Box::new(hir_cond),
                        then_body: vec![HirStmt::Return(Some(hir_value))],
                        then_expr: None,
                        else_body: vec![],
                        else_expr: None,
                    },
                    Type::Never,
                )
            }

            // Concurrency
            Expr::Go(body) => {
                let hir_body = self.lower_expr(body);
                let ty = Type::Handle(Box::new(hir_body.ty.clone()));
                HirExpr::new(HirExprKind::Go(Box::new(hir_body)), ty)
            }

            Expr::Receive(inner) => {
                let hir_inner = self.lower_expr(inner);
                let ty = match &hir_inner.ty {
                    Type::Channel(t) => *t.clone(),
                    Type::Handle(t) => *t.clone(),
                    _ => Type::Any,
                };
                HirExpr::new(HirExprKind::Receive(Box::new(hir_inner)), ty)
            }

            Expr::Send { chan, value } => {
                let hir_chan = self.lower_expr(chan);
                let hir_value = self.lower_expr(value);
                HirExpr::new(
                    HirExprKind::Send {
                        chan: Box::new(hir_chan),
                        value: Box::new(hir_value),
                    },
                    Type::Nil,
                )
            }

            Expr::Select(arms) => {
                let hir_arms: Vec<HirSelectArm> = arms
                    .iter()
                    .map(|arm| match arm {
                        SelectArm::Recv { var, chan, body } => {
                            let hir_chan = self.lower_expr(chan);
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Recv {
                                var: var.clone(),
                                chan: hir_chan,
                                body: hir_body,
                            }
                        }
                        SelectArm::Send { chan, value, body } => {
                            let hir_chan = self.lower_expr(chan);
                            let hir_value = self.lower_expr(value);
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Send {
                                chan: hir_chan,
                                value: hir_value,
                                body: hir_body,
                            }
                        }
                        SelectArm::Default(body) => {
                            let mut hir_body = Vec::new();
                            for s in body {
                                self.lower_stmt(s, &mut hir_body);
                            }
                            HirSelectArm::Default(hir_body)
                        }
                    })
                    .collect();
                HirExpr::new(HirExprKind::Select(hir_arms), Type::Any)
            }

            // Implicit self: `.field` → `__self.field` (or whatever self_param is set to)
            Expr::ImplicitSelf(field) => {
                let self_name = self
                    .self_param
                    .clone()
                    .unwrap_or_else(|| "__self".to_string());
                let target = HirExpr::new(HirExprKind::Ident(self_name), Type::Any);
                HirExpr::new(
                    HirExprKind::Member {
                        target: Box::new(target),
                        field: field.clone(),
                    },
                    Type::Any,
                )
            }

            // Prototype instantiation: Proto{k1:v1 k2:v2}
            // Desugars to: { _tmp = tok_map_clone(proto); tok_map_set(_tmp, "k1", v1); ...; _tmp }
            Expr::ProtoInit { proto, overrides } => self.lower_proto_init(proto, overrides),

            // Import as expression
            Expr::Import(path) => {
                // Import as expression returns a map
                HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_import".to_string(),
                        args: vec![HirExpr::new(HirExprKind::Str(path.clone()), Type::Str)],
                    },
                    Type::Map(Box::new(Type::Any)),
                )
            }

            // Return as expression
            Expr::Return(expr) => {
                let hir_expr = expr.as_ref().map(|e| self.lower_expr(e));
                // Emit return as a block with a return statement and Never result
                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![HirStmt::Return(hir_expr)],
                        expr: None,
                    },
                    Type::Never,
                )
            }

            Expr::Break => HirExpr::new(
                HirExprKind::Block {
                    stmts: vec![HirStmt::Break],
                    expr: None,
                },
                Type::Never,
            ),

            Expr::Continue => HirExpr::new(
                HirExprKind::Block {
                    stmts: vec![HirStmt::Continue],
                    expr: None,
                },
                Type::Never,
            ),
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Loop handling
    // ═══════════════════════════════════════════════════════════

    /// Define loop variables in scope based on the loop clause.
    pub(super) fn define_loop_vars(&mut self, clause: &LoopClause) {
        match clause {
            LoopClause::ForRange { var, .. } => {
                self.define_local(var, Type::Int);
            }
            LoopClause::ForEach { var, iter } => {
                let iter_ty = self.infer_expr_type(iter);
                let elem_ty = match iter_ty {
                    Type::Array(inner) => *inner,
                    Type::Str => Type::Str,
                    Type::Range => Type::Int,
                    _ => Type::Any,
                };
                self.define_local(var, elem_ty);
            }
            LoopClause::ForEachIndexed {
                idx_var,
                val_var,
                iter,
            } => {
                let iter_ty = self.infer_expr_type(iter);
                // For maps, idx is the string key; for arrays, idx is the integer index
                let idx_ty = if matches!(iter_ty, Type::Map(_)) {
                    Type::Str
                } else {
                    Type::Int
                };
                self.define_local(idx_var, idx_ty);
                let elem_ty = match iter_ty {
                    Type::Array(inner) => *inner,
                    Type::Map(inner) => *inner,
                    Type::Str => Type::Str,
                    Type::Range => Type::Int,
                    _ => Type::Any,
                };
                self.define_local(val_var, elem_ty);
            }
            LoopClause::While(_) | LoopClause::Infinite => {}
        }
    }

    /// Lower a loop expression.
    pub(super) fn lower_loop(&mut self, clause: &LoopClause, body: &LoopBody) -> HirExpr {
        match body {
            LoopBody::Block(stmts) => {
                let hir_kind = self.lower_loop_clause(clause);
                self.push_scope();
                self.define_loop_vars(clause);
                let mut hir_body = Vec::new();
                for s in stmts {
                    self.lower_stmt(s, &mut hir_body);
                }
                self.pop_scope();
                HirExpr::new(
                    HirExprKind::Loop {
                        kind: Box::new(hir_kind),
                        body: hir_body,
                    },
                    Type::Nil,
                )
            }
            // Desugar collect loop: ~(i:0..n)=expr ->
            //   { _collect = []; loop { _collect = push(_collect, expr) }; _collect }
            LoopBody::Collect(collect_expr) => {
                self.push_scope();
                self.define_loop_vars(clause);
                let hir_kind = self.lower_loop_clause(clause);
                let collect_tmp = self.gensym();
                let elem_ty = self.infer_expr_type(collect_expr);
                let arr_ty = Type::Array(Box::new(elem_ty));

                let hir_collect_expr = self.lower_expr(collect_expr);

                let loop_body = vec![HirStmt::Assign {
                    name: collect_tmp.clone(),
                    ty: arr_ty.clone(),
                    value: HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_push".to_string(),
                            args: vec![
                                HirExpr::new(
                                    HirExprKind::Ident(collect_tmp.clone()),
                                    arr_ty.clone(),
                                ),
                                hir_collect_expr,
                            ],
                        },
                        arr_ty.clone(),
                    ),
                }];

                let init_stmt = HirStmt::Assign {
                    name: collect_tmp.clone(),
                    ty: arr_ty.clone(),
                    value: HirExpr::new(
                        HirExprKind::RuntimeCall {
                            name: "tok_array_alloc".to_string(),
                            args: vec![],
                        },
                        arr_ty.clone(),
                    ),
                };

                let loop_expr = HirExpr::new(
                    HirExprKind::Loop {
                        kind: Box::new(hir_kind),
                        body: loop_body,
                    },
                    Type::Nil,
                );

                self.pop_scope();

                HirExpr::new(
                    HirExprKind::Block {
                        stmts: vec![init_stmt, HirStmt::Expr(loop_expr)],
                        expr: Some(Box::new(HirExpr::new(
                            HirExprKind::Ident(collect_tmp),
                            arr_ty.clone(),
                        ))),
                    },
                    arr_ty,
                )
            }
        }
    }

    fn lower_loop_clause(&mut self, clause: &LoopClause) -> HirLoopKind {
        match clause {
            LoopClause::While(cond) => {
                let hir_cond = self.lower_expr(cond);
                HirLoopKind::While(Box::new(hir_cond))
            }
            LoopClause::ForRange { var, range } => {
                // The range expression is an Expr::Range { start, end, inclusive }
                match range {
                    Expr::Range {
                        start,
                        end,
                        inclusive,
                    } => {
                        let hir_start = self.lower_expr(start);
                        let hir_end = self.lower_expr(end);
                        HirLoopKind::ForRange {
                            var: var.clone(),
                            start: hir_start,
                            end: hir_end,
                            inclusive: *inclusive,
                        }
                    }
                    // Fallback: treat as foreach over a general iterable
                    other => {
                        let hir_iter = self.lower_expr(other);
                        HirLoopKind::ForEach {
                            var: var.clone(),
                            iter: hir_iter,
                        }
                    }
                }
            }
            LoopClause::ForEach { var, iter } => {
                let hir_iter = self.lower_expr(iter);
                HirLoopKind::ForEach {
                    var: var.clone(),
                    iter: hir_iter,
                }
            }
            LoopClause::ForEachIndexed {
                idx_var,
                val_var,
                iter,
            } => {
                let hir_iter = self.lower_expr(iter);
                HirLoopKind::ForEachIndexed {
                    idx_var: idx_var.clone(),
                    val_var: val_var.clone(),
                    iter: hir_iter,
                }
            }
            LoopClause::Infinite => HirLoopKind::Infinite,
        }
    }

    pub(super) fn lower_block(&mut self, stmts: &[Stmt]) -> HirExpr {
        let mut hir_stmts = Vec::new();
        for s in stmts {
            self.lower_stmt(s, &mut hir_stmts);
        }
        // The last expression statement becomes the block's value
        let expr = if let Some(HirStmt::Expr(_)) = hir_stmts.last() {
            if let Some(HirStmt::Expr(e)) = hir_stmts.pop() {
                Some(Box::new(e))
            } else {
                None
            }
        } else {
            None
        };
        let ty = expr.as_ref().map(|e| e.ty.clone()).unwrap_or(Type::Nil);
        HirExpr::new(
            HirExprKind::Block {
                stmts: hir_stmts,
                expr,
            },
            ty,
        )
    }

    // ═══════════════════════════════════════════════════════════
    // Parameter and type helpers
    // ═══════════════════════════════════════════════════════════

    pub(super) fn lower_params(&self, params: &[Param]) -> Vec<HirParam> {
        params
            .iter()
            .map(|p| {
                let ty = if p.variadic {
                    // Variadic param collects remaining args into an array
                    Type::Array(Box::new(Type::Any))
                } else {
                    p.ty.as_ref()
                        .map(|te| self.resolve_type_expr(te))
                        .unwrap_or(Type::Any)
                };
                HirParam {
                    name: p.name.clone(),
                    ty,
                    variadic: p.variadic,
                    has_default: p.default.is_some(),
                }
            })
            .collect()
    }

    #[allow(clippy::only_used_in_recursion)]
    fn resolve_type_expr(&self, te: &ast::TypeExpr) -> Type {
        match te {
            ast::TypeExpr::Prim(p) => match p {
                ast::PrimType::Int => Type::Int,
                ast::PrimType::Float => Type::Float,
                ast::PrimType::Str => Type::Str,
                ast::PrimType::Bool => Type::Bool,
                ast::PrimType::Nil => Type::Nil,
                ast::PrimType::Any => Type::Any,
            },
            ast::TypeExpr::Array(inner) => Type::Array(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Map(inner) => Type::Map(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Tuple(elts) => {
                Type::Tuple(elts.iter().map(|e| self.resolve_type_expr(e)).collect())
            }
            ast::TypeExpr::Func(params, ret) => {
                let pts = params
                    .iter()
                    .map(|p| tok_types::ParamType {
                        ty: self.resolve_type_expr(p),
                        has_default: false,
                    })
                    .collect();
                Type::Func(tok_types::FuncType {
                    params: pts,
                    ret: Box::new(self.resolve_type_expr(ret)),
                    variadic: false,
                })
            }
            ast::TypeExpr::Optional(inner) => {
                Type::Optional(Box::new(self.resolve_type_expr(inner)))
            }
            ast::TypeExpr::Result(inner) => Type::Result(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Channel(inner) => Type::Channel(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Handle(inner) => Type::Handle(Box::new(self.resolve_type_expr(inner))),
            ast::TypeExpr::Var(_) => Type::Any,
        }
    }

    pub(super) fn lower_func_body(&mut self, body: &FuncBody) -> Vec<HirStmt> {
        match body {
            FuncBody::Expr(expr) => {
                let hir_expr = self.lower_expr(expr);
                vec![HirStmt::Return(Some(hir_expr))]
            }
            FuncBody::Block(stmts) => {
                let mut out = Vec::new();
                for s in stmts {
                    self.lower_stmt(s, &mut out);
                }
                out
            }
        }
    }
}
