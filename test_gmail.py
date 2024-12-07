# test_gmail.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

def test_gmail():
    email_config = {
       'sender': 'your_email@gmail.com',
       'smtp_server': 'smtp.gmail.com',
       'port': 465,
       'username': 'your_email@gmail.com',
       'password': 'generated_pin'
    }

    try:
        print(f"Testing Gmail connection for: {email_config['sender']}")
        with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['port']) as server:
            print("Connected to Gmail SMTP server")
            
            print("Attempting login...")
            server.login(email_config['username'], email_config['password'])
            print("Login successful!")
            
            # Create simple test message
            msg = MIMEMultipart()
            msg['Subject'] = "Test Email from Python"
            msg['From'] = email_config['sender']
            msg['To'] = email_config['sender']
            
            body = """
            <html>
                <body>
                    <h2>Test Email</h2>
                    <p>This is a test email sent at: {}</p>
                </body>
            </html>
            """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
            
            msg.attach(MIMEText(body, 'html'))
            
            print(f"Sending test email to: {msg['To']}")
            server.send_message(msg)
            print("Test email sent successfully!")
            
    except smtplib.SMTPAuthenticationError:
        print("Authentication failed! Please check your email and app password.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print("\nFull error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_gmail()
