use super::*;

impl<'a> Lowerer<'a> {
    /// Lower prototype instantiation: Proto{k1:v1 k2:v2}
    /// Desugars to: { _tmp = tok_map_clone(proto); tok_map_set(_tmp, "k1", v1); ...; _tmp }
    pub(super) fn lower_proto_init(&mut self, proto: &Expr, overrides: &[(MapKey, Expr)]) -> HirExpr {
        let tmp = self.gensym();
        let mut stmts = Vec::new();

        // _tmp = tok_map_clone(proto)
        let proto_hir = self.lower_expr(proto);
        let clone_call = HirExpr::new(
            HirExprKind::RuntimeCall {
                name: "tok_map_clone".to_string(),
                args: vec![proto_hir],
            },
            Type::Map(Box::new(Type::Any)),
        );
        stmts.push(HirStmt::Assign {
            name: tmp.clone(),
            ty: Type::Map(Box::new(Type::Any)),
            value: clone_call,
        });

        // Collect which overrides are methods (for method_fields tracking)
        let mut override_methods = std::collections::HashSet::new();

        // tok_map_set(_tmp, "key", value) for each override
        for (key, val) in overrides {
            let key_str = match key {
                MapKey::Ident(s) | MapKey::Str(s) => s.clone(),
            };
            let is_method = matches!(val, Expr::Lambda { .. });
            let val_hir = if is_method {
                // Inject __self as first param for methods
                self.lower_method_lambda(val)
            } else {
                self.lower_expr(val)
            };
            if is_method {
                override_methods.insert(key_str.clone());
            }
            let tmp_ident = HirExpr::new(
                HirExprKind::Ident(tmp.clone()),
                Type::Map(Box::new(Type::Any)),
            );
            let key_hir = HirExpr::new(HirExprKind::Str(key_str), Type::Str);
            stmts.push(HirStmt::Expr(HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_map_set".to_string(),
                    args: vec![tmp_ident, key_hir, val_hir],
                },
                Type::Nil,
            )));
        }

        // Merge method_fields from prototype with override methods
        if let Expr::Ident(proto_name) = proto {
            if let Some(proto_methods) = self.method_fields.get(proto_name).cloned() {
                let mut merged = proto_methods;
                merged.extend(override_methods);
                // We'll store this mapping after the assign (see lower_stmt)
                // For now, store under tmp so the assign handler can pick it up
                self.method_fields.insert(tmp.clone(), merged);
            } else if !override_methods.is_empty() {
                self.method_fields.insert(tmp.clone(), override_methods);
            }
        } else if !override_methods.is_empty() {
            self.method_fields.insert(tmp.clone(), override_methods);
        }

        // Final expression: _tmp
        let result = HirExpr::new(HirExprKind::Ident(tmp), Type::Map(Box::new(Type::Any)));
        HirExpr::new(
            HirExprKind::Block {
                stmts,
                expr: Some(Box::new(result)),
            },
            Type::Map(Box::new(Type::Any)),
        )
    }

    /// Lower a lambda that's a method in a map/prototype — inject `__self` as first param.
    pub(super) fn lower_method_lambda(&mut self, expr: &Expr) -> HirExpr {
        if let Expr::Lambda {
            params,
            ret_type: _,
            body,
        } = expr
        {
            // Inject __self as first parameter
            let mut new_params = vec![ast::Param {
                name: "__self".to_string(),
                ty: None,
                default: None,
                variadic: false,
            }];
            new_params.extend(params.clone());

            // Lower params
            let hir_params: Vec<HirParam> = new_params
                .iter()
                .map(|p| HirParam {
                    name: p.name.clone(),
                    ty: Type::Any,
                    variadic: p.variadic,
                    has_default: p.default.is_some(),
                })
                .collect();

            // Set self_param while lowering body
            let prev_self = self.self_param.take();
            self.self_param = Some("__self".to_string());
            self.push_scope();
            for p in &new_params {
                self.define_local(&p.name, Type::Any);
            }

            let hir_body = self.lower_func_body(body);

            self.pop_scope();
            self.self_param = prev_self;

            HirExpr::new(
                HirExprKind::Lambda {
                    params: hir_params,
                    ret_type: Type::Any,
                    body: hir_body,
                },
                Type::Func(tok_types::FuncType {
                    params: new_params
                        .iter()
                        .map(|_| tok_types::ParamType {
                            ty: Type::Any,
                            has_default: false,
                        })
                        .collect(),
                    ret: Box::new(Type::Any),
                    variadic: false,
                }),
            )
        } else {
            self.lower_expr(expr)
        }
    }

    /// Lower string interpolation to a chain of tok_string_concat runtime calls.
    pub(super) fn lower_interp(&mut self, parts: &[InterpPart]) -> HirExpr {
        let mut result: Option<HirExpr> = None;

        for part in parts {
            let segment = match part {
                InterpPart::Lit(s) => {
                    if s.is_empty() {
                        continue;
                    }
                    HirExpr::new(HirExprKind::Str(s.clone()), Type::Str)
                }
                InterpPart::Expr(e) => {
                    let hir_e = self.lower_expr(e);
                    // If the expression is already a string, use it directly;
                    // otherwise wrap with tok_value_to_string
                    if matches!(hir_e.ty, Type::Str) {
                        hir_e
                    } else {
                        // Choose a specific conversion based on known type
                        let conv_name = match &hir_e.ty {
                            Type::Int => "tok_int_to_string",
                            Type::Float => "tok_float_to_string",
                            Type::Bool => "tok_bool_to_string",
                            _ => "tok_value_to_string",
                        };
                        HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: conv_name.to_string(),
                                args: vec![hir_e],
                            },
                            Type::Str,
                        )
                    }
                }
            };

            result = Some(match result {
                None => segment,
                Some(acc) => HirExpr::new(
                    HirExprKind::RuntimeCall {
                        name: "tok_string_concat".to_string(),
                        args: vec![acc, segment],
                    },
                    Type::Str,
                ),
            });
        }

        // If all parts were empty, return empty string
        result.unwrap_or_else(|| HirExpr::new(HirExprKind::Str(String::new()), Type::Str))
    }

    /// Lower a function call that has spread arguments.
    ///
    /// Desugars `f(a ..arr b)` into:
    /// ```text
    /// { _tmp = alloc(); _tmp = push(_tmp, a); _tmp = concat(_tmp, arr);
    ///   _tmp = push(_tmp, b); f(_tmp[0], _tmp[1], ..., _tmp[N-1]) }
    /// ```
    /// For variadic functions: fixed args extracted by index, rest via slice.
    pub(super) fn lower_call_with_spread(&mut self, func: &Expr, args: &[Expr]) -> HirExpr {
        let hir_func = self.lower_expr(func);
        let arr_ty = Type::Array(Box::new(Type::Any));

        // Step 1: Build a flat array of all args (same pattern as array spread)
        let tmp = self.gensym();
        let mut stmts = Vec::new();

        // _tmp = tok_array_alloc()
        stmts.push(HirStmt::Assign {
            name: tmp.clone(),
            ty: arr_ty.clone(),
            value: HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_array_alloc".to_string(),
                    args: vec![],
                },
                arr_ty.clone(),
            ),
        });

        for arg in args {
            match arg {
                Expr::Spread(inner) => {
                    let hir_inner = self.lower_expr(inner);
                    // _tmp = tok_array_concat(_tmp, inner)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_concat".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_inner,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
                _ => {
                    let hir_arg = self.lower_expr(arg);
                    // _tmp = tok_array_push(_tmp, arg)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_push".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_arg,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
            }
        }

        // Step 2: Determine arity and extract args from the flat array
        let func_name = if let Expr::Ident(name) = func {
            Some(name.as_str())
        } else {
            None
        };

        let n_fixed = func_name.and_then(|n| self.func_variadic.get(n).copied());
        let arity = func_name.and_then(|n| self.func_arity.get(n).copied()).or(
            if let Type::Func(ft) = &hir_func.ty {
                Some(ft.params.len())
            } else {
                None
            },
        );

        let tmp_ident = || HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone());

        let mut hir_args = Vec::new();

        if let Some(n_fixed) = n_fixed {
            // Variadic function: extract fixed args, slice the rest
            for i in 0..n_fixed {
                hir_args.push(HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(i as i64), Type::Int)),
                    },
                    Type::Any,
                ));
            }
            // Variadic rest: tok_array_slice(_tmp, n_fixed, tok_array_len(_tmp))
            let len_expr = HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_array_len".to_string(),
                    args: vec![tmp_ident()],
                },
                Type::Int,
            );
            hir_args.push(HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_array_slice".to_string(),
                    args: vec![
                        tmp_ident(),
                        HirExpr::new(HirExprKind::Int(n_fixed as i64), Type::Int),
                        len_expr,
                    ],
                },
                arr_ty.clone(),
            ));
        } else if let Some(arity) = arity {
            // Non-variadic function with known arity: extract each arg by index
            for i in 0..arity {
                hir_args.push(HirExpr::new(
                    HirExprKind::Index {
                        target: Box::new(tmp_ident()),
                        index: Box::new(HirExpr::new(HirExprKind::Int(i as i64), Type::Int)),
                    },
                    Type::Any,
                ));
            }
        } else {
            // Unknown arity: pass the flat array as a single spread arg.
            // This works for variadic functions that take (..args).
            hir_args.push(tmp_ident());
        }

        let ret_ty = match &hir_func.ty {
            Type::Func(ft) => *ft.ret.clone(),
            _ => Type::Any,
        };

        let call_expr = HirExpr::new(
            HirExprKind::Call {
                func: Box::new(hir_func),
                args: hir_args,
            },
            ret_ty.clone(),
        );

        // Wrap in block: { _tmp = alloc(); ...; f(_tmp[0], ...) }
        HirExpr::new(
            HirExprKind::Block {
                stmts,
                expr: Some(Box::new(call_expr)),
            },
            ret_ty,
        )
    }

    /// Lower array literal, handling spread elements.
    pub(super) fn lower_array(&mut self, elts: &[Expr]) -> HirExpr {
        let has_spread = elts.iter().any(|e| matches!(e, Expr::Spread(_)));

        if !has_spread {
            // Simple case: no spreads
            let hir_elts: Vec<HirExpr> = elts.iter().map(|e| self.lower_expr(e)).collect();
            let elem_ty = if hir_elts.is_empty() {
                Type::Any
            } else {
                let mut t = hir_elts[0].ty.clone();
                for e in &hir_elts[1..] {
                    t = tok_types::unify(&t, &e.ty);
                }
                t
            };
            return HirExpr::new(HirExprKind::Array(hir_elts), Type::Array(Box::new(elem_ty)));
        }

        // Has spreads: desugar to alloc + concat/push chain
        let tmp = self.gensym();
        let arr_ty = Type::Array(Box::new(Type::Any));

        let mut stmts = Vec::new();
        // _tmp = tok_array_alloc()
        stmts.push(HirStmt::Assign {
            name: tmp.clone(),
            ty: arr_ty.clone(),
            value: HirExpr::new(
                HirExprKind::RuntimeCall {
                    name: "tok_array_alloc".to_string(),
                    args: vec![],
                },
                arr_ty.clone(),
            ),
        });

        for elt in elts {
            match elt {
                Expr::Spread(inner) => {
                    let hir_inner = self.lower_expr(inner);
                    // _tmp = tok_array_concat(_tmp, inner)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_concat".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_inner,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
                _ => {
                    let hir_elt = self.lower_expr(elt);
                    // _tmp = tok_array_push(_tmp, elt)
                    stmts.push(HirStmt::Assign {
                        name: tmp.clone(),
                        ty: arr_ty.clone(),
                        value: HirExpr::new(
                            HirExprKind::RuntimeCall {
                                name: "tok_array_push".to_string(),
                                args: vec![
                                    HirExpr::new(HirExprKind::Ident(tmp.clone()), arr_ty.clone()),
                                    hir_elt,
                                ],
                            },
                            arr_ty.clone(),
                        ),
                    });
                }
            }
        }

        HirExpr::new(
            HirExprKind::Block {
                stmts,
                expr: Some(Box::new(HirExpr::new(
                    HirExprKind::Ident(tmp),
                    arr_ty.clone(),
                ))),
            },
            arr_ty,
        )
    }

    /// Lower pipeline: x |> f -> f(x); x |> f(y) -> f(x, y)
    pub(super) fn lower_pipeline(&mut self, left: &Expr, right: &Expr) -> HirExpr {
        let hir_left = self.lower_expr(left);

        match right {
            // x |> f(y, z) -> f(x, y, z)
            Expr::Call { func, args } => {
                // If any arg is a spread, delegate to lower_call_with_spread
                // with left prepended to the args list
                if args.iter().any(|a| matches!(a, Expr::Spread(_))) {
                    let mut all_args = vec![left.clone()];
                    all_args.extend(args.iter().cloned());
                    return self.lower_call_with_spread(func, &all_args);
                }
                let hir_func = self.lower_expr(func);
                let mut hir_args = vec![hir_left];
                for arg in args {
                    hir_args.push(self.lower_expr(arg));
                }
                // Pad missing args with defaults (pipeline provides extra leading arg)
                if let Expr::Ident(name) = func.as_ref() {
                    let total_args = hir_args.len();
                    let missing: Vec<Expr> = self
                        .func_defaults
                        .get(name.as_str())
                        .into_iter()
                        .flat_map(|defaults| {
                            defaults.iter().skip(total_args).filter_map(|d| d.clone())
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
            // x |> f -> f(x)
            _ => {
                let hir_func = self.lower_expr(right);
                let mut hir_args = vec![hir_left];
                // Wrap arg into array for variadic functions with 0 fixed params
                if let Expr::Ident(name) = right {
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
        }
    }
}
