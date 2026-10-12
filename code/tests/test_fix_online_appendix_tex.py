"""Cross-document references become plain text in each standalone PDF's TeX."""

from fix_online_appendix_tex import repair, unlink_site

TEX = r"\label{sec:here} See Section~\ref{sec:here}, Section~\ref{sec:there} and Table~\ref{tab:x}."


def test_repair_unlinks_only_references_into_the_other_document():
    fixed, missing = repair(TEX, {"sec:there": "Online Appendix B"}, {"tab:x": "3"})
    assert fixed == (
        r"\label{sec:here} See Section~\ref{sec:here}, Online Appendix B and Table~3."
    )
    assert missing == []


def test_repair_reports_an_unresolvable_label():
    _, missing = repair(TEX, {}, {})
    assert missing == ["sec:there", "tab:x"]


def test_unlink_site_keeps_the_text_and_external_links():
    tex = r"\href{/online-appendix}{Online Appendix A} and \href{https://econ-ark.org/}{HARK}"
    assert unlink_site(tex) == (
        r"Online Appendix A and \href{https://econ-ark.org/}{HARK}",
        1,
    )
