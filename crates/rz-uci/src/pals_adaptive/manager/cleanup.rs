//! The root's original finite cleanup boundary. Installing this opt-in policy
//! does not grant a new search deadline or extend a caller's closure window.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct CleanupBoundary {
    original: Instant,
}

impl CleanupBoundary {
    pub(super) fn admit(
        search_deadline: Instant,
        original_cleanup_deadline: Instant,
    ) -> Result<Self, AdaptiveManagerError> {
        if original_cleanup_deadline < search_deadline {
            return Err(AdaptiveManagerError::Invalid(
                "cleanup boundary precedes original search deadline",
            ));
        }
        Ok(Self {
            original: original_cleanup_deadline,
        })
    }

    pub(super) fn original(self) -> Instant {
        self.original
    }

    pub(super) fn cap(self, requested: Instant) -> Instant {
        self.original.min(requested)
    }
}

pub(super) fn automatic_cleanup_deadline(
    original: Option<CleanupBoundary>,
) -> Result<Instant, AdaptiveManagerError> {
    let requested = Instant::now()
        .checked_add(Duration::from_secs(30))
        .ok_or(AdaptiveManagerError::Deadline)?;
    Ok(original.map_or(requested, |boundary| boundary.cap(requested)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_keeps_the_original_ceiling_for_explicit_and_automatic_closure() {
        let search = Instant::now();
        let original = search + Duration::from_secs(5);
        let boundary = CleanupBoundary::admit(search, original).unwrap();
        assert_eq!(boundary.original(), original);
        assert_eq!(boundary.cap(search + Duration::from_secs(60)), original);
        assert_eq!(boundary.cap(search), search);
        assert_eq!(
            automatic_cleanup_deadline(Some(boundary)).unwrap(),
            original
        );
        assert!(CleanupBoundary::admit(original, search).is_err());
    }
}
