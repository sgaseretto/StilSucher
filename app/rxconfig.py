import reflex as rx

config = rx.Config(
    app_name="app",
    db_url="sqlite:///semsearch.db",
    env=rx.Env.DEV,
)