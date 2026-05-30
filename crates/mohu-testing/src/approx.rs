// approx — implementation pending

pub fn assert_allclose(a: &[f32], b: &[f32]) {
    assert_allclose_tol(a, b, 1e-6, 1e-6);
}

pub fn assert_allclose_tol(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(a.len(), b.len(), "Lengths differ");
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        let threshold = atol + rtol * y.abs();
        assert!(
            diff <= threshold,
            "Values differ: {} vs {} (diff={}, threshold={})",
            x,
            y,
            diff,
            threshold
        );
    }
}
