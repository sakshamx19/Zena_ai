import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertTool:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        """
        Initialize the AlertTool with SMTP server details.
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def send_alert(self, message, recipient_email, llm_response=None):
        """
        Sends an alert via email with the given message and optional LLM response.
        """
        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = "Alert Notification"

            # Include the main message
            email_body = f"Message: {message}"
            
            # Append the LLM response if provided
            if llm_response:
                email_body += f"\n\nLLM Response:\n{llm_response}"
            
            msg.attach(MIMEText(email_body, 'plain'))

            # Connect to the SMTP server and send the email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print(f"ALERT sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send alert: {e}")
