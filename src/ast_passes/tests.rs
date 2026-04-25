use super::*;
use crate::ast::LoopKind;

#[test]
fn simplify_drops_self_assignment() {
    let stmt = Stmt::Assign {
        dst: LValue::Var("v0".to_string()),
        src: Expr::Reg("v0".to_string()),
    };
    assert_eq!(ast_simplify(stmt), Stmt::Empty);
}

#[test]
fn simplify_inverts_empty_then_if() {
    let stmt = Stmt::If {
        condition: Expr::Reg("b0".to_string()),
        then_branch: Box::new(Stmt::Empty),
        else_branch: Some(Box::new(Stmt::Return(None))),
    };
    assert_eq!(
        ast_simplify(stmt),
        Stmt::If {
            condition: Expr::Unary {
                op: "!".to_string(),
                arg: Box::new(Expr::Reg("b0".to_string())),
            },
            then_branch: Box::new(Stmt::Return(None)),
            else_branch: None,
        }
    );
}

#[test]
fn simplify_flattens_nested_sequences() {
    let stmt = Stmt::Sequence(vec![
        Stmt::Empty,
        Stmt::Sequence(vec![Stmt::ExprStmt(Expr::Reg("a".to_string()))]),
        Stmt::ExprStmt(Expr::Reg("b".to_string())),
    ]);
    assert_eq!(
        ast_simplify(stmt),
        Stmt::Sequence(vec![
            Stmt::ExprStmt(Expr::Reg("a".to_string())),
            Stmt::ExprStmt(Expr::Reg("b".to_string())),
        ])
    );
}

#[test]
fn simplify_preserves_loop_structure() {
    let stmt = Stmt::Loop {
        kind: LoopKind::While,
        condition: Some(Expr::Reg("b0".to_string())),
        body: Box::new(Stmt::Sequence(vec![Stmt::Empty, Stmt::Continue])),
    };
    assert_eq!(
        ast_simplify(stmt),
        Stmt::Loop {
            kind: LoopKind::While,
            condition: Some(Expr::Reg("b0".to_string())),
            body: Box::new(Stmt::Continue),
        }
    );
}
