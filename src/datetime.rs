//! The one module that names the date/time library (`jiff`, #383): every parse and
//! format of a date or instant goes through here, so the accepted forms and the
//! written forms are decided in one place.
//!
//! Everything is UTC and whole-library-feature-free (no tz database). `jiff`'s own
//! parsers are more lenient than the `chrono` ones they replaced (ISO 8601 basic
//! `20260301`, a missing seconds field, a `[Zone]` annotation…), and several callers
//! depend on those being refused — `changes_since`'s `since` must not read a
//! commit-id-shaped `20260301` as a date, and a schema's `timestamp` type is RFC 3339
//! — so the parsers here check the shape first and hand only a well-formed text to
//! `jiff`.

use jiff::{
    Timestamp, Unit,
    civil::{Date, DateTime},
    tz::TimeZone,
};

/// An RFC 3339 `date-time`: `YYYY-MM-DD`, `T` (or `t`, or a space), `HH:MM:SS`, an
/// optional `.fraction`, then `Z`/`z` or a `±HH:MM` offset — and nothing else. The
/// instant it names, or `None`.
pub(crate) fn parse_rfc3339(text: &str) -> Option<Timestamp> {
    if !rfc3339_shape(text.as_bytes()) {
        return None;
    }
    text.parse::<Timestamp>().ok()
}

/// The exact RFC 3339 `date-time` grammar's shape (the calendar and range checks are
/// `jiff`'s).
fn rfc3339_shape(b: &[u8]) -> bool {
    let digits = |range: std::ops::Range<usize>| {
        b.get(range)
            .is_some_and(|s| s.iter().all(u8::is_ascii_digit))
    };
    if !(digits(0..4)
        && b.get(4) == Some(&b'-')
        && digits(5..7)
        && b.get(7) == Some(&b'-')
        && digits(8..10)
        && matches!(b.get(10), Some(b'T' | b't' | b' '))
        && digits(11..13)
        && b.get(13) == Some(&b':')
        && digits(14..16)
        && b.get(16) == Some(&b':')
        && digits(17..19))
    {
        return false;
    }
    let mut i = 19;
    if b.get(i) == Some(&b'.') {
        let start = i + 1;
        i = start;
        while b.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    match &b[i..] {
        [b'Z' | b'z'] => true,
        [b'+' | b'-', ..] => {
            b.len() == i + 6 && digits(i + 1..i + 3) && b[i + 3] == b':' && digits(i + 4..i + 6)
        }
        _ => false,
    }
}

/// A calendar date written `YYYY-MM-DD` (a four-digit year; month and day may be one
/// or two digits, as the former parser read them), or `None`. Never an ISO 8601 basic
/// `YYYYMMDD` or a date with a time after it.
pub(crate) fn parse_date(text: &str) -> Option<Date> {
    let mut parts = text.split('-');
    let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let digits = |s: &str, lens: std::ops::RangeInclusive<usize>| {
        lens.contains(&s.len()) && s.bytes().all(|c| c.is_ascii_digit())
    };
    if !(digits(year, 4..=4) && digits(month, 1..=2) && digits(day, 1..=2)) {
        return None;
    }
    Date::new(year.parse().ok()?, month.parse().ok()?, day.parse().ok()?).ok()
}

/// Midnight UTC at the start of `date`.
pub(crate) fn midnight_utc(date: Date) -> Option<Timestamp> {
    TimeZone::UTC.to_timestamp(date.at(0, 0, 0, 0)).ok()
}

/// `YYYY-MM-DD` as midnight UTC, or `None`.
pub(crate) fn parse_date_midnight(text: &str) -> Option<Timestamp> {
    parse_date(text).and_then(midnight_utc)
}

/// A date and time without an offset, read as UTC, in one of `formats` (strptime
/// syntax); the first that matches wins.
pub(crate) fn parse_civil_utc(text: &str, formats: &[&str]) -> Option<Timestamp> {
    formats.iter().find_map(|fmt| {
        let dt = DateTime::strptime(fmt, text).ok()?;
        TimeZone::UTC.to_timestamp(dt).ok()
    })
}

/// Whole seconds since the Unix epoch, rounded toward negative infinity — what
/// `chrono`'s `timestamp()` returned (`jiff`'s `as_second` truncates toward zero, so a
/// pre-1970 instant with a fraction would otherwise land one second late).
pub(crate) fn floor_seconds(ts: Timestamp) -> i64 {
    ts.as_second() - i64::from(ts.subsec_nanosecond() < 0)
}

/// Seconds since the Unix epoch with the fraction kept.
pub(crate) fn epoch_seconds_f64(ts: Timestamp) -> f64 {
    ts.as_second() as f64 + f64::from(ts.subsec_nanosecond()) / 1e9
}

/// `ts` rounded to the microsecond and written as RFC 3339 UTC with a `Z`: no
/// fraction for whole seconds, otherwise three or six digits. That fixed width is what
/// the Qdrant payload has always held (`.500Z`, not `.5Z`), so a stored value is
/// byte-identical across the library change and no payload refresh is triggered.
pub(crate) fn payload_text(ts: Timestamp) -> Option<String> {
    let rounded = ts
        .round(jiff::TimestampRound::new().smallest(Unit::Microsecond))
        .ok()?;
    let dt = TimeZone::UTC.to_datetime(rounded);
    let nanos = dt.subsec_nanosecond();
    let fraction = if nanos == 0 {
        String::new()
    } else if nanos % 1_000_000 == 0 {
        format!(".{:03}", nanos / 1_000_000)
    } else {
        format!(".{:06}", nanos / 1_000)
    };
    Some(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{fraction}Z",
        dt.year(),
        dt.month(),
        dt.day(),
        dt.hour(),
        dt.minute(),
        dt.second()
    ))
}

/// Epoch seconds as RFC 3339 UTC (`1970-01-01T00:00:00Z`), or `None` out of range.
pub(crate) fn format_rfc3339_secs(secs: i64) -> Option<String> {
    Timestamp::from_second(secs).ok().map(|t| t.to_string())
}

/// Epoch seconds as the UTC calendar date `YYYY-MM-DD`, or `None` out of range.
pub(crate) fn format_date(secs: i64) -> Option<String> {
    Timestamp::from_second(secs)
        .ok()
        .map(|t| TimeZone::UTC.to_datetime(t).date().to_string())
}

/// Epoch seconds as `YYYY-MM-DDTHH:MMZ` (UTC, minute precision) — the form a
/// document's history and `changes_since` report, and `changes_since`'s `since` reads
/// back — or `None` out of range.
pub(crate) fn format_minute(secs: i64) -> Option<String> {
    Timestamp::from_second(secs)
        .ok()
        .map(|t| t.strftime("%Y-%m-%dT%H:%MZ").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(text: &str) -> Option<i64> {
        parse_rfc3339(text).map(floor_seconds)
    }

    #[test]
    fn rfc3339_accepts_the_grammar_and_normalizes_offsets() {
        assert_eq!(secs("2024-01-15T00:00:00Z"), Some(1_705_276_800));
        assert_eq!(secs("2024-01-15T01:00:00+01:00"), Some(1_705_276_800));
        assert_eq!(secs("2024-01-15t00:00:00z"), Some(1_705_276_800));
        assert_eq!(secs("2024-01-15 00:00:00Z"), Some(1_705_276_800));
        assert_eq!(secs("2024-01-15T00:00:00.5Z"), Some(1_705_276_800));
    }

    #[test]
    fn rfc3339_refuses_what_the_former_parser_refused() {
        for bad in [
            "",
            "2024-01-15",
            "20240115T000000Z",
            "2024-01-15T00:00Z",
            "2024-01-15T00Z",
            "2024-01-15T00:00:00",
            "2024-01-15T00:00:00+0100",
            "2024-01-15T00:00:00+01",
            "2024-01-15T00:00:00.Z",
            "2024-01-15T00:00:00Z[UTC]",
            "2024-01-15T00:00:00Z ",
            "2024-13-15T00:00:00Z",
            "2024-02-30T00:00:00Z",
            "soon",
        ] {
            assert!(parse_rfc3339(bad).is_none(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn dates_are_strict_year_month_day() {
        assert!(parse_date("2026-07-31").is_some());
        // The former parser took unpadded months and days; existing documents may carry them.
        assert!(parse_date("2024-1-5").is_some());
        for bad in [
            "",
            "20260731",
            "2026-07-31T12:00:00",
            "2026-07-31 ",
            "31/07/2026",
            "2024-13-01",
            "2024-02-30",
            "24-01-01",
            "2024-001-01",
            "+2024-01-01",
            "2024--1-01",
        ] {
            assert!(parse_date(bad).is_none(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn a_date_is_midnight_utc() {
        assert_eq!(
            parse_date_midnight("2024-01-15").map(floor_seconds),
            Some(1_705_276_800)
        );
    }

    #[test]
    fn floor_seconds_rounds_toward_negative_infinity() {
        let before_epoch = Timestamp::new(-1, 500_000_000).unwrap(); // -0.5 s
        assert_eq!(before_epoch.as_second(), 0);
        assert_eq!(floor_seconds(before_epoch), -1);
        assert_eq!(floor_seconds(Timestamp::new(5, 500_000_000).unwrap()), 5);
        assert!((epoch_seconds_f64(before_epoch) + 0.5).abs() < 1e-9);
    }

    #[test]
    fn payload_text_has_a_fixed_width_fraction() {
        let t = |s: &str| payload_text(parse_rfc3339(s).unwrap()).unwrap();
        assert_eq!(t("2024-01-02T00:00:00Z"), "2024-01-02T00:00:00Z");
        assert_eq!(t("2024-01-02T00:00:00.5Z"), "2024-01-02T00:00:00.500Z");
        assert_eq!(t("2024-01-02T00:00:00.12Z"), "2024-01-02T00:00:00.120Z");
        assert_eq!(
            t("2024-01-02T00:00:00.1234Z"),
            "2024-01-02T00:00:00.123400Z"
        );
        assert_eq!(
            t("2024-01-02T10:00:00.123456789+02:00"),
            "2024-01-02T08:00:00.123457Z"
        );
        // Rounding carries into the next second.
        assert_eq!(t("2024-01-02T00:00:00.9999996Z"), "2024-01-02T00:00:01Z");
        // And before the epoch the fraction still reads as time past the second.
        assert_eq!(t("1969-12-31T23:59:59.5Z"), "1969-12-31T23:59:59.500Z");
    }

    #[test]
    fn formatters_render_utc_and_refuse_out_of_range() {
        assert_eq!(
            format_rfc3339_secs(0).as_deref(),
            Some("1970-01-01T00:00:00Z")
        );
        assert_eq!(
            format_rfc3339_secs(1_700_000_000).as_deref(),
            Some("2023-11-14T22:13:20Z")
        );
        assert_eq!(format_date(1_700_000_000).as_deref(), Some("2023-11-14"));
        assert_eq!(
            format_minute(1_700_000_000).as_deref(),
            Some("2023-11-14T22:13Z")
        );
        assert_eq!(format_rfc3339_secs(i64::MAX), None);
        assert_eq!(format_date(i64::MAX), None);
        assert_eq!(format_minute(i64::MAX), None);
    }

    #[test]
    fn civil_datetimes_are_read_as_utc_in_the_listed_forms() {
        let forms = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"];
        assert_eq!(
            parse_civil_utc("2026-03-01 12:30", &forms).map(floor_seconds),
            Some(1_772_368_200)
        );
        assert!(parse_civil_utc("2026-03-01 12:30:00Z", &forms).is_none());
        assert!(parse_civil_utc("20260301", &forms).is_none());
    }
}
