//! Logical persistent-root registration for an exclusive engine owner.
//! This module does not attest CPU/GPU physical completion. UCI's actual native
//! fence, worker joins and shutdown receipt must precede owner transfer here.
//! Store authority uses the UCI-issued process-monotonic root, not game-local +1.

use super::archive_lifecycle::allocation_controls;
use super::*;
use crate::pals::store::{ArchiveOwnerSnapshot, ArchiveStats, StoreReadOwner};
use rz_contracts::pals::SearchAuthority;
use std::sync::{Arc, Weak};

#[path = "no_researcher_lifecycle.rs"]
mod no_researcher_lifecycle;
#[path = "preparing_root/registration.rs"]
mod preparing_registration;

#[derive(Default)]
pub struct AdaptiveRootRegistry {
    identity: Arc<()>,
    last: Option<RegisteredRoot>,
    pending_nonce: Option<u64>,
    next_nonce: u64,
    new_game_ready: Option<SearchAuthority>,
}
struct RegisteredRoot {
    authority: SearchAuthority,
    owner: StoreReadOwner,
    generation: u64,
    root: SituationId,
    snapshot: PositionSnapshot,
}
/// First-go readiness belongs to the original reset, not a registered root.
/// Setup must preserve both this registry and the still-zero Store generation.
pub(super) struct UnusedArchiveSetupSeal {
    registry: Weak<()>,
    store: StoreReadOwner,
    scope: SearchAuthority,
    next_nonce: u64,
}
impl UnusedArchiveSetupSeal {
    pub(super) fn validate_engine_game_preparation<M: RoleModel>(
        &self,
        semantic: &PalsSemanticState<M>,
    ) -> Result<(), PalsError> {
        semantic.ensure_adaptive_logical_quiescence()?;
        semantic.stores.validate_unused_archive_setup()?;
        let registry = &semantic.adaptive_root_registry;
        if !self.registry.ptr_eq(&semantic.adaptive_registry_identity())
            || !self.store.matches(&semantic.stores)
            || semantic.game_generation != Some(self.scope.game.0)
            || registry.last.is_some()
            || registry.pending_nonce.is_some()
            || registry.next_nonce != 0
            || self.next_nonce != 0
            || registry
                .new_game_ready
                .is_some_and(|ready| ready != self.scope)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(())
    }
    pub(super) fn validate<M: RoleModel>(
        &self,
        semantic: &PalsSemanticState<M>,
    ) -> Result<(), PalsError> {
        semantic.ensure_adaptive_logical_quiescence()?;
        let registry = &semantic.adaptive_root_registry;
        if !self.registry.ptr_eq(&semantic.adaptive_registry_identity())
            || !self.store.matches(&semantic.stores)
            || semantic.stores.generation() != 0
            || semantic.stores.root().is_some()
            || semantic.game_generation != Some(self.scope.game.0)
            || registry.last.is_some()
            || registry.pending_nonce.is_some()
            || registry.next_nonce != self.next_nonce
            || registry.new_game_ready != Some(self.scope)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(())
    }
}
/// NonClone and tied to one engine registry and its exact live Store. It proves
/// a checked logical root registration only, never a physical release fence.
pub struct AdaptiveRootRegistration {
    registry: Weak<()>,
    nonce: u64,
    authority: SearchAuthority,
    owner: StoreReadOwner,
    generation: u64,
    root: SituationId,
    snapshot: PositionSnapshot,
}

/// Private, move-only permission for one already idle archive-game transition.
/// It binds the old Store and unspent registry state, not merely a next tuple.
pub(super) struct IdleArchiveGameSeal {
    registry: Weak<()>,
    store: StoreReadOwner,
    generation: u64,
    game: u64,
    last: Option<SearchAuthority>,
    nonce: u64,
    next: SearchAuthority,
    archive: ArchiveOwnerSnapshot,
}
impl IdleArchiveGameSeal {
    pub(super) fn validate<M: RoleModel>(
        &self,
        semantic: &PalsSemanticState<M>,
    ) -> Result<(), PalsError> {
        semantic.ensure_adaptive_logical_quiescence()?;
        if !self.registry.ptr_eq(&semantic.adaptive_registry_identity())
            || !self.store.matches(&semantic.stores)
            || semantic.stores.generation() != self.generation
            || semantic.game_generation != Some(self.game)
            || semantic.last_adaptive_authority() != self.last
            || semantic.adaptive_root_registry.next_nonce != self.nonce
            || semantic.adaptive_root_registry.pending_nonce.is_some()
        {
            return Err(StoreError::StaleConsumer.into());
        }
        Ok(())
    }
}

impl<M: RoleModel> PalsSemanticState<M> {
    pub(super) fn seal_unused_engine_game_preparation(
        &self,
        scope: SearchAuthority,
    ) -> Result<UnusedArchiveSetupSeal, PalsError> {
        let seal = UnusedArchiveSetupSeal {
            registry: self.adaptive_registry_identity(),
            store: self.stores.read_owner(),
            scope,
            next_nonce: self.adaptive_root_registry.next_nonce,
        };
        seal.validate_engine_game_preparation(self)?;
        Ok(seal)
    }
    pub(super) fn commit_unused_engine_game_preparation(
        &mut self,
        scope: SearchAuthority,
    ) -> Result<(), PalsError> {
        self.seal_unused_engine_game_preparation(scope)?;
        self.adaptive_root_registry.new_game_ready = Some(scope);
        Ok(())
    }
    pub(super) fn seal_unused_archive_setup(
        &self,
        scope: SearchAuthority,
    ) -> Result<UnusedArchiveSetupSeal, PalsError> {
        if self.adaptive_root_registry.next_nonce != 0 {
            return Err(StoreError::StaleConsumer.into());
        }
        let seal = UnusedArchiveSetupSeal {
            registry: self.adaptive_registry_identity(),
            store: self.stores.read_owner(),
            scope,
            next_nonce: self.adaptive_root_registry.next_nonce,
        };
        seal.validate(self)?;
        Ok(seal)
    }

    pub(super) fn validate_idle_new_game_target(
        &self,
        next: SearchAuthority,
    ) -> Result<IdleArchiveGameSeal, PalsError> {
        self.ensure_archive_admission_open()?;
        self.ensure_adaptive_logical_quiescence()?;
        self.validate_adaptive_namespace(next)?;
        let game = self.game_generation.ok_or(StoreError::StaleConsumer)?;
        if next.game.0 <= game || self.adaptive_root_registry.pending_nonce.is_some() {
            return Err(StoreError::StaleConsumer.into());
        }
        let archive = self
            .stores
            .archive_owner_snapshot()?
            .ok_or(StoreError::ArchiveDisabled)?;
        if archive.admission_closed || !archive.complete {
            return Err(
                StoreError::InvalidConditions("idle newgame archive owner is not open").into(),
            );
        }
        Ok(IdleArchiveGameSeal {
            registry: self.adaptive_registry_identity(),
            store: self.stores.read_owner(),
            generation: self.stores.generation(),
            game,
            last: self.last_adaptive_authority(),
            nonce: self.adaptive_root_registry.next_nonce,
            next,
            archive,
        })
    }

    pub(super) fn commit_idle_new_game_scope(
        &mut self,
        seal: &IdleArchiveGameSeal,
        replacement: &mut Option<Box<PalsStores>>,
    ) -> Result<(), PalsError> {
        seal.validate(self)?;
        let previous =
            self.archive_state
                .closed_current_owner
                .ok_or(StoreError::InvalidConditions(
                    "idle newgame original close missing",
                ))?;
        let next_store = replacement.as_ref().ok_or(StoreError::InvalidConditions(
            "idle newgame original replacement missing",
        ))?;
        let next = next_store
            .archive_owner_snapshot()?
            .ok_or(StoreError::ArchiveDisabled)?;
        if previous.owner_id != seal.archive.owner_id
            || !previous.admission_closed
            || !previous.complete
            || next.owner_id == previous.owner_id
            || next.admission_closed
            || !next.complete
            || next.root_sha256 != previous.root_sha256
            || next.repository_sha256 != previous.repository_sha256
            || next.runtime_limits != previous.runtime_limits
            || next.session_write_ledger_id != previous.session_write_ledger_id
            || next.session_write_bytes_max != previous.session_write_bytes_max
            || next.session_write_bytes_consumed != previous.session_write_bytes_consumed
            || next.write_bytes_reserved != 0
            || next_store.generation() != 0
            || next_store.root().is_some()
            || next_store.hot_stats() != ArchiveStats::default()
            || seal.store.matches(next_store)
        {
            return Err(StoreError::InvalidConditions(
                "idle newgame original Store/ledger transfer",
            )
            .into());
        }
        // A model callback can panic. Keep both actual Stores in their owner
        // slots until it returns; all later publication steps are infallible.
        self.model.new_game_with_generation(Some(seal.next.game.0));
        self.publish_idle_archive_game(
            seal.next,
            replacement.take().expect("validated original replacement"),
            previous,
        );
        self.adaptive_root_registry.new_game_ready = Some(seal.next);
        Ok(())
    }

    pub(super) fn adaptive_registry_identity(&self) -> Weak<()> {
        Arc::downgrade(&self.adaptive_root_registry.identity)
    }

    pub(super) fn adaptive_new_game_scope(&self) -> Option<SearchAuthority> {
        self.adaptive_root_registry.new_game_ready
    }

    pub(super) fn last_adaptive_authority(&self) -> Option<SearchAuthority> {
        self.adaptive_root_registry
            .last
            .as_ref()
            .map(|last| last.authority)
    }

    pub(super) fn adaptive_archive_authority(
        &self,
        next: SearchAuthority,
    ) -> Result<SearchAuthority, PalsError> {
        self.validate_adaptive_namespace(next)?;
        let game = self.game_generation.ok_or(StoreError::StaleConsumer)?;
        Ok(SearchAuthority {
            game: rz_contracts::GameGeneration(game),
            root: rz_contracts::RootGeneration(self.stores.generation()),
            ..next
        })
    }

    pub(super) fn validate_adaptive_namespace(
        &self,
        next: SearchAuthority,
    ) -> Result<(), PalsError> {
        if let Some(last) = &self.adaptive_root_registry.last {
            if next.epoch != last.authority.epoch
                || next.implementation != last.authority.implementation
                || next.game.0 < last.authority.game.0
                || next.root.0 <= last.authority.root.0
            {
                return Err(StoreError::StaleConsumer.into());
            }
        }
        Ok(())
    }

    pub(super) fn register_adaptive_root(
        &mut self,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<AdaptiveRootRegistration, PalsError> {
        self.ensure_archive_admission_open()?;
        self.ensure_adaptive_logical_quiescence()?;
        self.validate_adaptive_namespace(authority)?;
        if self.adaptive_root_registry.pending_nonce.is_some()
            || self.game_generation != Some(authority.game.0)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        let snapshot = position.snapshot();
        let mut attested_retirement = false;
        if let Some(last) = &self.adaptive_root_registry.last {
            if last.authority.game == authority.game {
                if !last.owner.matches(&self.stores) {
                    return Err(StoreError::StaleConsumer.into());
                }
                if self.stores.generation() == last.generation {
                    if self.stores.root() != Some(last.root)
                        || !self
                            .stores
                            .states
                            .get(self.stores.situations.get(last.root)?.state)?
                            .same_state(&last.snapshot)
                    {
                        return Err(StoreError::StaleConsumer.into());
                    }
                } else {
                    attested_retirement = self.stores.root().is_none()
                        && last.generation.checked_add(1) == Some(self.stores.generation())
                        && self.last_engine_archive().is_some_and(|receipt| {
                            receipt.actual_root().is_some_and(|(root, _, generation)| {
                                root == last.root && generation == last.generation
                            })
                        });
                    if !attested_retirement {
                        return Err(StoreError::StaleConsumer.into());
                    }
                }
            } else {
                let ready = self
                    .adaptive_root_registry
                    .new_game_ready
                    .ok_or(StoreError::StaleConsumer)?;
                if ready.epoch != authority.epoch
                    || ready.implementation != authority.implementation
                    || ready.game != authority.game
                    || authority.root.0 < ready.root.0
                    || last.owner.matches(&self.stores)
                    || self.stores.generation() != 0
                    || self.stores.root().is_some()
                {
                    return Err(StoreError::StaleConsumer.into());
                }
            }
        }
        let nonce = self.adaptive_root_registry.next_nonce;
        let next_nonce = nonce.checked_add(1).ok_or(StoreError::RevisionExhausted)?;
        let registration = if attested_retirement && authority.root.0 == self.stores.generation() {
            self.stores.prepare_hot_root_registration_after_retirement(
                snapshot.clone(),
                authority.root.0,
            )?
        } else {
            self.stores
                .prepare_hot_root_registration(snapshot.clone(), authority.root.0)?
        };
        let root = self.stores.hot_focus_registered_root(registration)?;
        self.adaptive_root_registry.next_nonce = next_nonce;
        self.adaptive_root_registry.pending_nonce = Some(nonce);
        // Root authority was published by the successful Store commit even if
        // later port construction is canceled. Keep its monotonic history.
        self.adaptive_root_registry.last = Some(RegisteredRoot {
            authority,
            owner: self.stores.read_owner(),
            generation: self.stores.generation(),
            root,
            snapshot: snapshot.clone(),
        });
        self.adaptive_root_registry.new_game_ready = None;
        Ok(AdaptiveRootRegistration {
            registry: Arc::downgrade(&self.adaptive_root_registry.identity),
            nonce,
            authority,
            owner: self.stores.read_owner(),
            generation: self.stores.generation(),
            root,
            snapshot,
        })
    }

    pub(super) fn consume_adaptive_root_registration(
        &mut self,
        registration: AdaptiveRootRegistration,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<SituationId, PalsError> {
        self.ensure_adaptive_logical_quiescence()?;
        if !registration
            .registry
            .ptr_eq(&Arc::downgrade(&self.adaptive_root_registry.identity))
            || self.adaptive_root_registry.pending_nonce != Some(registration.nonce)
            || registration.authority != authority
            || !registration.owner.matches(&self.stores)
            || registration.generation != authority.root.0
            || self.stores.generation() != registration.generation
            || self.stores.root() != Some(registration.root)
            || self.game_generation != Some(authority.game.0)
            || !registration.snapshot.same_state(&position.snapshot())
            || !self
                .stores
                .states
                .get(self.stores.situations.get(registration.root)?.state)?
                .same_state(&registration.snapshot)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.adaptive_root_registry.pending_nonce = None;
        Ok(registration.root)
    }

    pub(super) fn abandon_unconsumed_adaptive_root(
        &mut self,
        authority: SearchAuthority,
    ) -> Result<(), PalsError> {
        self.ensure_adaptive_logical_quiescence()?;
        let last = self
            .adaptive_root_registry
            .last
            .as_ref()
            .ok_or(StoreError::StaleConsumer)?;
        if self.adaptive_root_registry.pending_nonce.is_none()
            || last.authority != authority
            || !last.owner.matches(&self.stores)
            || self.stores.generation() != last.generation
            || self.stores.root() != Some(last.root)
            || !self
                .stores
                .states
                .get(self.stores.situations.get(last.root)?.state)?
                .same_state(&last.snapshot)
        {
            return Err(StoreError::StaleConsumer.into());
        }
        self.adaptive_root_registry.pending_nonce = None;
        Ok(())
    }

    pub(super) fn ensure_adaptive_logical_quiescence(&self) -> Result<(), PalsError> {
        self.ensure_frontier_retired()?;
        if self.stores.tasks.active_states().next().is_some()
            || self.stores.review_pin_count()? != 0
            || self.nodes.iter().any(|node| node.resume.is_some())
        {
            return Err(StoreError::InvalidConditions(
                "adaptive root still owns live logical work",
            )
            .into());
        }
        Ok(())
    }
}

impl<M: RoleModel> PcSemanticOwner<M> {
    pub fn last_adaptive_authority(&self) -> Option<SearchAuthority> {
        self.semantic.last_adaptive_authority()
    }
    pub fn register_adaptive_root(
        &mut self,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<AdaptiveRootRegistration, PalsError> {
        self.semantic.register_adaptive_root(position, authority)
    }
    pub fn consume_adaptive_root_registration(
        &mut self,
        registration: AdaptiveRootRegistration,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<SituationId, PalsError> {
        self.semantic
            .consume_adaptive_root_registration(registration, position, authority)
    }
    pub fn abandon_unconsumed_adaptive_root(
        &mut self,
        authority: SearchAuthority,
    ) -> Result<(), PalsError> {
        self.semantic.abandon_unconsumed_adaptive_root(authority)
    }
}

impl<M: RoleModel> PalsEngine<M> {
    pub(super) fn adaptive_registry_identity(&self) -> Weak<()> {
        self.semantic.adaptive_registry_identity()
    }
    pub(super) fn adaptive_new_game_scope(&self) -> Option<SearchAuthority> {
        self.semantic.adaptive_new_game_scope()
    }
    pub fn last_adaptive_authority(&self) -> Option<SearchAuthority> {
        self.semantic.last_adaptive_authority()
    }

    /// Current maintenance envelope for this exact Store, independently of the
    /// later search root. It carries no authority to admit a neural question.
    pub fn adaptive_archive_authority(
        &self,
        next: SearchAuthority,
    ) -> Result<SearchAuthority, PalsError> {
        self.semantic.adaptive_archive_authority(next)
    }

    pub(super) fn ensure_adaptive_logical_quiescence(&self) -> Result<(), PalsError> {
        self.semantic.ensure_adaptive_logical_quiescence()?;
        if self.held_cpu_task.is_some() {
            return Err(StoreError::InvalidConditions(
                "adaptive root still owns live logical work",
            )
            .into());
        }
        Ok(())
    }

    fn validate_adaptive_namespace(&self, next: SearchAuthority) -> Result<(), PalsError> {
        self.semantic.validate_adaptive_namespace(next)
    }

    /// Register the exact new Rules root with its actual UCI-issued generation.
    /// Skipped accepted Go/Position commands are ordinary monotonic gaps.
    pub fn register_adaptive_root(
        &mut self,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<AdaptiveRootRegistration, PalsError> {
        self.ensure_archive_admission_open()?;
        self.ensure_adaptive_logical_quiescence()?;
        self.semantic.register_adaptive_root(position, authority)
    }

    pub fn consume_adaptive_root_registration(
        &mut self,
        registration: AdaptiveRootRegistration,
        position: &Position,
        authority: SearchAuthority,
    ) -> Result<SituationId, PalsError> {
        self.ensure_adaptive_logical_quiescence()?;
        self.semantic
            .consume_adaptive_root_registration(registration, position, authority)
    }

    /// Cancel an unpublished port admission while retaining the already checked
    /// Store root and its monotonic scope. This revokes any outstanding proof;
    /// it does not certify native completion or roll back a root generation.
    pub fn abandon_unconsumed_adaptive_root(
        &mut self,
        authority: SearchAuthority,
    ) -> Result<(), PalsError> {
        self.ensure_adaptive_logical_quiescence()?;
        self.semantic.abandon_unconsumed_adaptive_root(authority)
    }

    /// Private-worker operation. Existing official game reset performs checker,
    /// model and archive lifecycle work; the manager must not call this inline.
    /// One reset also handles skipped UCI newgame commands by explicitly binding
    /// their final issued game number after the successful reset.
    pub fn try_adaptive_new_game(
        &mut self,
        next: SearchAuthority,
        deadline: Instant,
        cancel: &AtomicBool,
    ) -> Result<(), PalsError> {
        self.ensure_adaptive_logical_quiescence()?;
        self.validate_adaptive_namespace(next)?;
        let previous_game = self
            .semantic
            .game_generation
            .ok_or(StoreError::StaleConsumer)?;
        if next.game.0 <= previous_game
            || self.semantic.adaptive_root_registry.pending_nonce.is_some()
        {
            return Err(StoreError::StaleConsumer.into());
        }
        allocation_controls(Some((deadline, cancel)))?;
        self.try_new_game(deadline, cancel)?;
        if self.semantic.game_generation != previous_game.checked_add(1)
            || self.semantic.stores.root().is_some()
            || self.semantic.stores.generation() != 0
            || !self.semantic.nodes.is_empty()
            || !self.semantic.records.is_empty()
        {
            return Err(
                StoreError::InvalidConditions("official adaptive newgame reset differs").into(),
            );
        }
        self.ensure_adaptive_logical_quiescence()?;
        // This callback runs on the lifecycle worker, with actual UCI fences
        // already checked. It cannot be mixed into a manager semantic commit.
        if next.game.0 != previous_game + 1 {
            self.semantic
                .model
                .new_game_with_generation(Some(next.game.0));
        }
        self.semantic.game_generation = Some(next.game.0);
        self.semantic.adaptive_root_registry.new_game_ready = Some(next);
        allocation_controls(Some((deadline, cancel)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuConfig;
    use rz_contracts::{Digest, GameGeneration, ProcessEpoch, RootGeneration};

    fn engine() -> PalsEngine<LegalOrderRoleMock> {
        PalsEngine::new(
            PalsConfig {
                beam_width: 1,
                line_plies: 2,
                max_nodes: 257,
                max_records: 32,
                max_role_calls: 64,
                cpu_nodes_per_task: 256,
            },
            LegalOrderRoleMock,
            CpuEngine::new(CpuConfig {
                tt_entries: 64,
                max_depth: 4,
                quiescence_ply: 4,
                ..CpuConfig::default()
            })
            .unwrap(),
        )
        .unwrap()
    }
    fn scope(game: u64, root: u64) -> SearchAuthority {
        SearchAuthority {
            epoch: ProcessEpoch(2),
            game: GameGeneration(game),
            root: RootGeneration(root),
            implementation: Digest([3; 32]),
        }
    }
    fn accept(
        engine: &mut PalsEngine<LegalOrderRoleMock>,
        position: &Position,
        authority: SearchAuthority,
    ) -> SituationId {
        let proof = engine.register_adaptive_root(position, authority).unwrap();
        engine
            .consume_adaptive_root_registration(proof, position, authority)
            .unwrap()
    }
    #[test]
    fn skipped_position_and_go_numbers_register_actual_monotonic_root() {
        let mut engine = engine();
        let mut position = Position::startpos();
        accept(&mut engine, &position, scope(0, 2));
        position.make_move("e2e4".parse().unwrap()).unwrap();
        accept(&mut engine, &position, scope(0, 7));
        assert_eq!(engine.semantic.stores.generation(), 7);
        assert_eq!(engine.last_adaptive_authority(), Some(scope(0, 7)));
        assert!(matches!(
            engine.register_adaptive_root(&position, scope(0, 7)),
            Err(PalsError::Store(StoreError::StaleConsumer))
        ));
    }
    #[test]
    fn repeated_go_keeps_exact_situation_but_revokes_old_generation() {
        let mut engine = engine();
        let position = Position::startpos();
        let first = accept(&mut engine, &position, scope(0, 1));
        let next = accept(&mut engine, &position, scope(0, 4));
        assert_eq!(first, next);
        assert_eq!(engine.semantic.stores.generation(), 4);
    }
    #[test]
    fn wrong_epoch_implementation_and_history_do_not_consume_registration() {
        let mut engine = engine();
        let position = Position::startpos();
        accept(&mut engine, &position, scope(0, 1));
        let mut wrong = scope(0, 3);
        wrong.epoch = ProcessEpoch(99);
        assert!(engine.register_adaptive_root(&position, wrong).is_err());
        wrong = scope(0, 3);
        wrong.implementation = Digest([99; 32]);
        assert!(engine.register_adaptive_root(&position, wrong).is_err());
        let proof = engine
            .register_adaptive_root(&position, scope(0, 3))
            .unwrap();
        let fen_only = Position::from_fen(&position.to_fen()).unwrap();
        assert!(!position.snapshot().same_state(&fen_only.snapshot()));
        assert!(
            engine
                .consume_adaptive_root_registration(proof, &fen_only, scope(0, 3))
                .is_err()
        );
        engine
            .abandon_unconsumed_adaptive_root(scope(0, 3))
            .unwrap();
        accept(&mut engine, &position, scope(0, 5));
    }
    #[test]
    fn discarded_proof_is_revoked_without_rolling_back_registered_root() {
        let mut engine = engine();
        let position = Position::startpos();
        let abandoned = engine
            .register_adaptive_root(&position, scope(0, 4))
            .unwrap();
        engine
            .abandon_unconsumed_adaptive_root(scope(0, 4))
            .unwrap();
        accept(&mut engine, &position, scope(0, 9));
        assert!(
            engine
                .consume_adaptive_root_registration(abandoned, &position, scope(0, 4))
                .is_err()
        );
        assert_eq!(engine.semantic.stores.generation(), 9);
    }
    #[test]
    fn official_newgame_preserves_process_root_counter_and_binds_skipped_games() {
        let mut engine = engine();
        let position = Position::startpos();
        accept(&mut engine, &position, scope(0, 6));
        let old_owner = engine.semantic.stores.read_owner();
        engine
            .try_adaptive_new_game(
                scope(3, 12),
                Instant::now() + Duration::from_secs(5),
                &AtomicBool::new(false),
            )
            .unwrap();
        assert_eq!(engine.semantic.game_generation, Some(3));
        assert!(!old_owner.matches(&engine.semantic.stores));
        assert_eq!(engine.last_adaptive_authority(), Some(scope(0, 6)));
        accept(&mut engine, &position, scope(3, 15));
        assert_eq!(engine.semantic.stores.generation(), 15);
        assert_eq!(engine.last_adaptive_authority(), Some(scope(3, 15)));
    }
    #[test]
    fn legacy_store_replacement_cannot_impersonate_official_newgame_binding() {
        let mut engine = engine();
        let position = Position::startpos();
        accept(&mut engine, &position, scope(0, 2));
        engine.new_game();
        assert!(matches!(
            engine.register_adaptive_root(&position, scope(1, 6)),
            Err(PalsError::Store(StoreError::StaleConsumer))
        ));
    }
    #[test]
    fn active_review_pin_blocks_root_transition_until_explicit_release() {
        let mut engine = engine();
        let position = Position::startpos();
        let root = accept(&mut engine, &position, scope(0, 2));
        let mut pins = crate::pals::store::StorePins::default();
        pins.situations.insert(root);
        let held = engine.semantic.stores.pin_review(pins).unwrap();
        assert!(
            engine
                .register_adaptive_root(&position, scope(0, 4))
                .is_err()
        );
        assert_eq!(engine.semantic.stores.generation(), 2);
        drop(held);
        accept(&mut engine, &position, scope(0, 4));
    }
}
