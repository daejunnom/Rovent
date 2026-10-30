//! Pure authored accounting fixtures. No directory, process or tensor work.
use super::*;
fn h(c: char) -> String {
    c.to_string().repeat(64)
}
fn counts(p: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("proposer".into(), p),
        ("critic".into(), 0),
        ("validator".into(), 0),
    ])
}
fn update(n: u64, input: char, origin: char) -> Value {
    json!({"event":{"role":"proposer","completed_role_updates":n,
    "sample_inputs":vec![h(input);8],"sample_origins":vec![h(origin);8]}})
}
#[test]
fn older_snapshot_does_not_inherit_later_consumed_samples() {
    let updates = vec![update(1, 'a', 'b'), update(2, 'c', 'd')];
    let (inputs, origins) = prefix_sets(
        counts(0),
        &updates,
        BTreeSet::from([h('1')]),
        BTreeSet::from([h('2')]),
        &counts(1),
    )
    .unwrap();
    assert_eq!(inputs, BTreeSet::from([h('1'), h('a')]));
    assert_eq!(origins, BTreeSet::from([h('2'), h('b')]));
    assert!(!inputs.contains(&h('c')));
    assert!(!origins.contains(&h('d')));
}
#[test]
fn zero_new_update_snapshot_retains_exact_parent_sets() {
    let inputs = BTreeSet::from([h('1')]);
    let origins = BTreeSet::from([h('2')]);
    let actual = prefix_sets(
        counts(3),
        &[update(4, 'a', 'b')],
        inputs.clone(),
        origins.clone(),
        &counts(3),
    )
    .unwrap();
    assert_eq!(actual, (inputs, origins));
}
#[test]
fn missing_boundary_bad_sequence_and_foreign_roles_are_rejected() {
    assert!(
        prefix_sets(
            counts(0),
            &[update(1, 'a', 'b')],
            BTreeSet::new(),
            BTreeSet::new(),
            &counts(2)
        )
        .is_err()
    );
    assert!(
        prefix_sets(
            counts(0),
            &[update(2, 'a', 'b')],
            BTreeSet::new(),
            BTreeSet::new(),
            &counts(2)
        )
        .is_err()
    );
    let mut foreign = update(1, 'a', 'b');
    foreign["event"]["role"] = json!("foreign");
    assert!(
        prefix_sets(
            counts(0),
            &[foreign],
            BTreeSet::new(),
            BTreeSet::new(),
            &counts(1)
        )
        .is_err()
    );
}
#[test]
fn tail_has_all_eight_actual_differences_and_no_refund() {
    let before = COST_FIELDS
        .map(|k| (k.into(), 20))
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let after = COST_FIELDS
        .map(|k| (k, 25))
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let result = actual_tail_delta(&before, &json!(after)).unwrap();
    assert_eq!(result.len(), 8);
    assert!(result.values().all(|n| *n == 5));
    let mut bad = json!(after);
    bad["partial_accumulation_ns"] = json!(19);
    assert!(actual_tail_delta(&before, &bad).is_err());
    bad = json!(after);
    bad.as_object_mut().unwrap().remove("replay_ns");
    assert!(actual_tail_delta(&before, &bad).is_err());
}
