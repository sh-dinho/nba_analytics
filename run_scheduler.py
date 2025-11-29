import schedule
import time
from predict_picks import send_daily_picks

schedule.every().day.at("10:00").do(send_daily_picks)

while True:
    schedule.run_pending()
    time.sleep(60)
