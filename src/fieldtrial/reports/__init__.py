__all__ = ["planning_calendar_payload", "render_analysis_report", "render_planning_report"]


def __getattr__(name: str):
    if name == "render_analysis_report":
        from fieldtrial.reports.analysis import render_analysis_report

        return render_analysis_report
    if name == "render_planning_report":
        from fieldtrial.reports.planning import render_planning_report

        return render_planning_report
    if name == "planning_calendar_payload":
        from fieldtrial.reports.visuals import planning_calendar_payload

        return planning_calendar_payload
    raise AttributeError(name)
