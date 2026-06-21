import os

def update_template(filepath, active_link):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the top part to replace
    start_body = content.find('<body>')
    if start_body == -1: return
    end_sidebar = content.find('</section>')
    
    # We just want to replace the head and sidebar.
    # Let's do a simpler regex or split:
    
    # Standard header
    header = f"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
    <meta charset="UTF-8">
    <title>Mental Health Assessment</title>
    <link rel="stylesheet" href="{{{{ url_for('static',filename='style.css') }}}}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/css/all.min.css" />
    <link href="https://unpkg.com/boxicons@2.0.7/css/boxicons.min.css" rel="stylesheet" />
</head>
<body>
    <div class="side_bar">
        <div class="profile">
            <div class="logo">Mental Health Assessment</div>
        </div>
        <ul>
            <li><a {'class="active"' if active_link == 'home' else ''} href="home"><i class="fas fa-qrcode"></i>Home</a></li>
            <li><a {'class="active"' if active_link == 'eda' else ''} href="eda"><i class="fas fa-chart-bar"></i>EDA</a></li>
            <li><a {'class="active"' if active_link == 'detector' else ''} href="detector"><i class="fas fa-heartbeat"></i>Diagnosis</a></li>
            <li><a {'class="active"' if active_link == 'model_parameter' else ''} href="model_parameter"><i class="fas fa-cogs"></i>Model Parameters</a></li>
        </ul>
    </div>
"""

    # We need to find where <section> starts.
    section_start = content.find('<section')
    if section_start != -1:
        section_end = content.find('>', section_start) + 1
        inner_content = content[section_end:content.rfind('</section>')]
        
        # Inject glass class to first div if possible or wrap it
        # Actually, let's just wrap inner_content
        new_section = f"""
    <section>
        <div class="glass content_card">
            {inner_content}
        </div>
    </section>
"""
    else:
        new_section = ""

    # Replace the chatbot at the bottom
    footer = """
    {% include 'chatbot.html' %}
</body>
</html>
"""

    # For detector.html which has scripts at the bottom:
    script_part = ""
    script_start = content.rfind('<script>')
    if script_start != -1 and 'dis_ptsd' in content[script_start:]:
        script_part = content[script_start:content.rfind('</body>')]

    final_content = header + new_section + script_part + footer

    with open(filepath, 'w') as f:
        f.write(final_content)

update_template('templates/eda.html', 'eda')
update_template('templates/model_parameter.html', 'model_parameter')
update_template('templates/detector.html', 'detector')

print("Templates updated successfully!")
