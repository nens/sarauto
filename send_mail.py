import smtplib, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import config

def send_email(log_name, subject, body):
    filename = log_name + '.log'
    email_user = config.email_user
    pswd = config.pswd
    send_to = config.send_to
    email_send = send_to.split(",")
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = ','.join(email_send)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    part = MIMEBase('application', "octet-stream")
    part.set_payload( open(filename,"rb").read() )
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(filename))
    msg.attach(part)
    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email_user, pswd)
    server.sendmail(email_user, email_send, text)
    server.quit()