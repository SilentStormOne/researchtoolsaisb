from flask import Flask, render_template_string, jsonify
import os
import time

app = Flask(__name__)

# Track the last known modification time
last_known_mod_time = 0


@app.route("/")
def index():
    return render_template_string(
        """
<!DOCTYPE html>
<html>
<head>
    <title>Debug Log Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #333; /* Dark gray background */
            color: #fff; /* White text color */
            font-family: 'Fira Code', monospace;
        }
        pre {
            white-space: pre-wrap; /* Enable word wrapping */
            word-wrap: break-word;
        }
        #updateNotification {
            display: none;
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #444;
            color: #0f0;
            padding: 8px;
            border-radius: 5px;
            opacity: 1;
            transition: opacity 2s; /* Fade effect duration */
        }
    </style>
    <script>
        function refreshContent() {
            fetch("/content")
                .then(response => response.text())
                .then(data => {
                    document.getElementById("logContent").textContent = data;
                })
                .catch(console.error);
        }

        function checkForUpdate() {
            fetch("/check-update")
                .then(response => response.json())
                .then(data => {
                    if(data.updated) {
                        const updateBox = document.getElementById("updateNotification");
                        updateBox.style.display = "block";
                        updateBox.style.opacity = 1; // Reset opacity in case it's been faded out
                        setTimeout(() => {
                            updateBox.style.opacity = 0; // Start fading out after 7.5 seconds
                        }, 7500);
                        setTimeout(() => {
                            updateBox.style.display = "none"; // Hide after fade out
                        }, 9500); // Wait for fade out to finish
                    }
                })
                .catch(console.error);
        }

        setInterval(refreshContent, 2000); // Refresh content every 2 seconds
        setInterval(checkForUpdate, 2000); // Check for updates every 2 seconds
        window.onload = () => {
            refreshContent();
            checkForUpdate();
        };
    </script>
</head>
<body>
    <h1>Debug Log</h1>
    <div id="updateNotification">Updated.</div>
    <pre id="logContent">Loading...</pre>
</body>
</html>
"""
    )


@app.route("/content")
def content():
    try:
        with open("logs/debug.txt", "r", encoding="utf-8", errors="replace") as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return str(e)


@app.route("/check-update")
def check_update():
    global last_known_mod_time
    try:
        current_mod_time = os.path.getmtime("logs/debug.txt")
        if current_mod_time != last_known_mod_time:
            last_known_mod_time = current_mod_time
            return jsonify({"updated": True})
        return jsonify({"updated": False})
    except FileNotFoundError:
        return jsonify({"updated": False, "error": "File not found."})
    except Exception as e:
        return jsonify({"updated": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
