if __name__ == "__main__":
    
    email_config = {
       'sender': 'your_email@gmail.com',
       'smtp_server': 'smtp.gmail.com',
       'port': 465,
       'username': 'your_email@gmail.com',
       'password': 'generated_pin'
    }

    recipient_emails = ['your_email@gmail.com', 'friends_email', 'more_friends@email']


    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup-service':
            install_required_packages()
            setup_background_service()
        elif sys.argv[1] == '--install-packages':
            install_required_packages()
        elif sys.argv[1] == '--test-report':
            try:
                print("Testing weekly report generation...")
                alert_system = EnhancedCryptoAlertSystem(email_config, recipient_emails)
                report = alert_system.generate_weekly_report()
                alert_system.send_email(
                    subject="Crypto Weekly Report - Test",
                    body=report
                )
                print("Test report generated and sent successfully!")
            except Exception as e:
                print(f"Error testing report: {str(e)}")
                traceback.print_exc()
        elif sys.argv[1] == '--test-email':
            try:
                alert_system = EnhancedCryptoAlertSystem(email_config, recipient_emails)
                alert_system.send_email(
                    subject="Crypto Alert System - Test Email",
                    body="<html><body><h2>Test Email</h2><p>System is working!</p></body></html>"
                )
                print("Test email sent successfully!")
            except Exception as e:
                print(f"Error sending test email: {str(e)}")
    else:
        try:
            print("Starting Crypto Alert System...")
            alert_system = EnhancedCryptoAlertSystem(email_config, recipient_emails)
            alert_system.run_scheduled_tasks()
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Error running alert system: {str(e)}")
            traceback.print_exc()
