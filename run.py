from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.notify import send_daily_picks
from app import create_app
import logging

app = create_app()

def schedule_daily_task():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_daily_picks, 'interval', hours=24)  # Runs once every 24 hours
    scheduler.start()

if __name__ == "__main__":
    schedule_daily_task()
    app.run(debug=True, host="0.0.0.0", port=5000)
