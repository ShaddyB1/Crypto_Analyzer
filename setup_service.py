#!/usr/bin/env python3
import os
import sys
import subprocess

def setup_service():
    """Set up the crypto alert service with background process handling"""
    try:
        # Get paths
        home_dir = os.path.expanduser('~')
        script_dir = os.getcwd()
        script_path = os.path.join(script_dir, 'crypto_alert_system.py')
        
        # Create launch agent plist content
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.cryptoalert</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/anaconda3/bin/python3</string>
        <string>{script_path}</string>
        <string>--run</string>
    </array>
    <key>UserName</key>
    <string>{os.environ.get('USER')}</string>
    <key>WorkingDirectory</key>
    <string>{script_dir}</string>
    <key>StandardOutPath</key>
    <string>{script_dir}/crypto_alert.log</string>
    <key>StandardErrorPath</key>
    <string>{script_dir}/crypto_alert_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONPATH</key>
        <string>/opt/anaconda3/lib/python3.11/site-packages</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ProcessType</key>
    <string>Background</string>
    <key>Nice</key>
    <integer>1</integer>
</dict>
</plist>
"""
        
        # Create LaunchAgents directory if it doesn't exist
        launch_agents_dir = os.path.join(home_dir, 'Library/LaunchAgents')
        os.makedirs(launch_agents_dir, exist_ok=True)
        
        # Write plist file
        plist_path = os.path.join(launch_agents_dir, 'com.user.cryptoalert.plist')
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        
        # Set permissions
        os.chmod(plist_path, 0o644)
        os.chmod(script_path, 0o755)
        
        # Clean up any existing service
        subprocess.run(['launchctl', 'bootout', f"gui/{os.getuid()}", plist_path], capture_output=True)
        
        # Load the new service
        subprocess.run(['launchctl', 'bootstrap', f"gui/{os.getuid()}", plist_path], check=True)
        
        print("\nCrypto Alert Service installed successfully!")
        print(f"\nService identifier: com.user.cryptoalert")
        print("\nTo monitor:")
        print(f"- View logs: tail -f {script_dir}/crypto_alert.log")
        print(f"- View errors: tail -f {script_dir}/crypto_alert_error.log")
        print(f"- Check status: launchctl list | grep cryptoalert")
        
    except Exception as e:
        print(f"Error setting up service: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    setup_service()