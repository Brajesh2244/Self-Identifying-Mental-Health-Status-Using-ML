from flask import *
import joblib
import os, sys
import pandas as pd  # for dataframe handling
import numpy as np  # for matrices
import matplotlib.pyplot as plt  # for plotting graphs
import plotly.graph_objects as go  # for gauge meter plots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report  # evaluation metrics
import seaborn as sns  # for plotting confusion matrix
import random  # for randomly displaying quotes

app = Flask(__name__)

user_id = 'admin'
user_pwd = 'admin123'

print(os.getlogin())


@app.route('/')
def log():
    return render_template('login.html')


@app.route('/submit_log', methods=['GET', 'POST'])
def log_sub():
    if request.method == 'POST':
        uid = request.form['uid']
        upwd = request.form['pwd']
        error = 'Invalid Credentials'

        if uid == user_id and upwd == user_pwd:
            return render_template('index.html')
        else:
            return render_template('login.html', error=error)


@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/eda')
def eda():
    return render_template('eda.html')


@app.route('/detector')
def detector():
    return render_template('detector.html')


@app.route('/model_parameter')
def model_parameter():
    return render_template('model_parameter.html')


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static_new', filename=filename))


@app.route('/submit_d', methods=['GET', 'POST'])
def submit():
    global stacking_classifier_proba
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        country = request.form['country']
        self_employed = request.form['self_employed']
        family_history = request.form['family_history']
        treatment = request.form['treatment']
        work_interfere = request.form['work_interfere']
        remote_work = request.form['remote_work']
        tech_company = request.form['tech_company']
        benefit_available = request.form['benefit_available']
        care_options = request.form['care_options']
        wellness_program = request.form['wellness_program']
        seek_help = request.form['seek_help']
        leave = request.form['leave']
        mental_health_consequence = request.form['mental_health_consequence']
        physical_health_consequence = request.form['physical_health_consequence']
        supervisor = request.form['supervisor']
        mental_health_interview = request.form['mental_health_interview']
        physical_health_interview = request.form['physical_health_interview']
        mental_vs_physical = request.form['mental_vs_physical']
        file_path = os.path.abspath('static/uploads/stable.jpg')
        # LOADING & PROCESSING DATA
        df = pd.read_csv(
            r"psychological_instability_data.csv")  # loading the raw_dataset

        df = df.drop(columns=['Timestamp', 'comments', 'no_employees', 'anonymity', 'coworkers',
                              'state'])  # dropping unnecessary columns

        x = df["self_employed"].mode()[0]  # replacing NaN values with mode
        df["self_employed"].fillna(x, inplace=True)
        y = df["work_interfere"].mode()[0]  # replacing NaN values with mode
        df["work_interfere"].fillna(y, inplace=True)

        male_str = ['M', 'Male', 'male', 'm', 'Male-ish', 'something kinda male?',
                    'Mal', 'Male (CIS)', 'male leaning androgynous', 'cis male',
                    'Cis Male', 'msle', 'p', 'Mail', 'Make', 'Malr', 'maile']
        female_str = ['Female', "Female", 'female', 'Cis Female', 'Woman', 'F',
                      'f', 'Femake', 'woman', 'cis-female/femme', 'femail']

        # create dictionaries for mapping
        male_dict = {key: 0 for key in male_str}
        female_dict = {key: 1 for key in female_str}

        # map keys from male_str to 0 in dataframe
        df['Gender'] = df['Gender'].map(male_dict).fillna(df['Gender'])
        # print(df)
        # map keys from female_str to 1 in dataframe
        df['Gender'] = df['Gender'].map(female_dict).fillna(df['Gender'])
        # print(df)
        # map remaining keys from complete_list to 2 in dataframe
        df['Gender'] = df['Gender'].apply(lambda x: 2 if x not in [0, 1] else x)
        # print(df)

        # print(df.columns[df.isna().any()])

        a1 = np.arange(1, len(df['Country'].unique().tolist()) + 1,
                       1)  # creating a numpy array of length = number of unique elements
        b1 = df['Country'].unique().tolist()  # creating a list of unique entries in the given column
        c1 = {j: i for i, j in zip(a1, b1)}  # creating a dictionary with a1 and b1
        df['Country'] = df['Country'].map(c1)  # mapping the dictionary to dataframe

        a2 = np.arange(1, len(df['self_employed'].unique().tolist()) + 1, 1)
        b2 = df['self_employed'].unique().tolist()
        c2 = {j: i for i, j in zip(a2, b2)}
        df['self_employed'] = df['self_employed'].map(c2)

        a3 = np.arange(1, len(df['family_history'].unique().tolist()) + 1, 1)
        b3 = df['family_history'].unique().tolist()
        c3 = {j: i for i, j in zip(a3, b3)}
        df['family_history'] = df['family_history'].map(c3)

        a4 = np.arange(1, len(df['treatment'].unique().tolist()) + 1, 1)
        b4 = df['treatment'].unique().tolist()
        c4 = {j: i for i, j in zip(a4, b4)}
        df['treatment'] = df['treatment'].map(c4)

        a5 = np.arange(1, len(df['work_interfere'].unique().tolist()) + 1, 1)
        b5 = df['work_interfere'].unique().tolist()
        c5 = {j: i for i, j in zip(a5, b5)}
        df['work_interfere'] = df['work_interfere'].map(c5)

        a6 = np.arange(1, len(df['remote_work'].unique().tolist()) + 1, 1)
        b6 = df['remote_work'].unique().tolist()
        c6 = {j: i for i, j in zip(a6, b6)}
        df['remote_work'] = df['remote_work'].map(c6)

        a7 = np.arange(1, len(df['tech_company'].unique().tolist()) + 1, 1)
        b7 = df['tech_company'].unique().tolist()
        c7 = {j: i for i, j in zip(a7, b7)}
        df['tech_company'] = df['tech_company'].map(c7)

        a8 = np.arange(1, len(df['benefits'].unique().tolist()) + 1, 1)
        b8 = df['benefits'].unique().tolist()
        c8 = {j: i for i, j in zip(a8, b8)}
        df['benefits'] = df['benefits'].map(c8)

        a9 = np.arange(1, len(df['care_options'].unique().tolist()) + 1, 1)
        b9 = df['care_options'].unique().tolist()
        c9 = {j: i for i, j in zip(a9, b9)}
        df['care_options'] = df['care_options'].map(c9)

        a10 = np.arange(1, len(df['wellness_program'].unique().tolist()) + 1, 1)
        b10 = df['wellness_program'].unique().tolist()
        c10 = {j: i for i, j in zip(a10, b10)}
        df['wellness_program'] = df['wellness_program'].map(c10)

        a11 = np.arange(1, len(df['seek_help'].unique().tolist()) + 1, 1)
        b11 = df['seek_help'].unique().tolist()
        c11 = {j: i for i, j in zip(a11, b11)}
        df['seek_help'] = df['seek_help'].map(c11)

        a12 = np.arange(1, len(df['leave'].unique().tolist()) + 1, 1)
        b12 = df['leave'].unique().tolist()
        c12 = {j: i for i, j in zip(a12, b12)}
        df['leave'] = df['leave'].map(c12)

        a13 = np.arange(1, len(df['mental_health_consequence'].unique().tolist()) + 1, 1)
        b13 = df['mental_health_consequence'].unique().tolist()
        c13 = {j: i for i, j in zip(a13, b13)}
        df['mental_health_consequence'] = df['mental_health_consequence'].map(c13)

        a14 = np.arange(1, len(df['phys_health_consequence'].unique().tolist()) + 1, 1)
        b14 = df['phys_health_consequence'].unique().tolist()
        c14 = {j: i for i, j in zip(a14, b14)}
        df['phys_health_consequence'] = df['phys_health_consequence'].map(c14)

        a15 = np.arange(1, len(df['supervisor'].unique().tolist()) + 1, 1)
        b15 = df['supervisor'].unique().tolist()
        c15 = {j: i for i, j in zip(a15, b15)}
        df['supervisor'] = df['supervisor'].map(c15)

        a16 = np.arange(1, len(df['mental_health_interview'].unique().tolist()) + 1, 1)
        b16 = df['mental_health_interview'].unique().tolist()
        c16 = {j: i for i, j in zip(a16, b16)}
        df['mental_health_interview'] = df['mental_health_interview'].map(c16)

        a17 = np.arange(1, len(df['phys_health_interview'].unique().tolist()) + 1, 1)
        b17 = df['phys_health_interview'].unique().tolist()
        c17 = {j: i for i, j in zip(a17, b17)}
        df['phys_health_interview'] = df['phys_health_interview'].map(c17)

        a18 = np.arange(1, len(df['mental_vs_physical'].unique().tolist()) + 1, 1)
        b18 = df['mental_vs_physical'].unique().tolist()
        c18 = {j: i for i, j in zip(a18, b18)}
        df['mental_vs_physical'] = df['mental_vs_physical'].map(c18)

        print(df['obs_consequence'].value_counts())

        sc = joblib.load("scaler.pkl")

        # binod these are the inputs
        Age = age

        Gender = gender
        gender_digit = {"MALE": 0, "FEMALE": 1, "TRANSGENDER": 2}
        Gender = gender_digit[Gender]

        Country = country
        Country = c1[Country]

        self_employed = self_employed
        self_employed = c2[self_employed]

        family_history = family_history
        family_history = c3[family_history]

        treatment = treatment
        treatment = c4[treatment]

        work_interfere = work_interfere
        work_interfere = c5[work_interfere]

        remote_work = remote_work
        remote_work = c6[remote_work]

        tech_company = tech_company
        tech_company = c7[tech_company]

        benefits = benefit_available
        benefits = c8[benefits]

        care_options = care_options
        care_options = c9[care_options]

        wellness_program = wellness_program
        wellness_program = c10[wellness_program]

        seek_help = seek_help
        seek_help = c11[seek_help]

        leave = leave
        leave = c12[leave]

        mental_health_consequence = mental_health_consequence
        mental_health_consequence = c13[mental_health_consequence]

        phys_health_consequence = physical_health_consequence
        phys_health_consequence = c14[phys_health_consequence]

        supervisor = supervisor
        supervisor = c15[supervisor]

        mental_health_interview = mental_health_interview
        mental_health_interview = c16[mental_health_interview]

        phys_health_interview = physical_health_interview
        phys_health_interview = c17[phys_health_interview]

        mental_vs_physical = mental_vs_physical
        mental_vs_physical = c18[mental_vs_physical]

        new_user_input = [[Age, Gender, Country, self_employed, family_history, treatment,
                           work_interfere, remote_work, tech_company, benefits,
                           care_options, wellness_program, seek_help, leave,
                           mental_health_consequence, phys_health_consequence, supervisor,
                           mental_health_interview, phys_health_interview,
                           mental_vs_physical]]

        labels = ["stable", "unstable"]

        # TRAINING HYBRID MODEL
        stacking_classifier = joblib.load('stacked_model.pkl')
        stacking_classifier_res = stacking_classifier.predict(sc.transform(new_user_input))
        stacking_classifier_proba = stacking_classifier.predict_proba(sc.transform(new_user_input))

        # Create the Plotly figure for stable class
        stable_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stacking_classifier_proba[0][0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of class stable"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "green"}
            }
        ))
        unstable_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stacking_classifier_proba[0][1],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probability of class unstable"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "red"}
            }
        ))

        # Convert the Plotly figure to HTML
        stable_html = stable_fig.to_html(full_html=False)

        unstable_html = unstable_fig.to_html(full_html=False)

        # getting the value from unstable hidden section

        # Pass the HTML to the template for rendering
        return render_template('detector.html', unstable_html=unstable_html, stable_html=stable_html,
                               op=stacking_classifier_proba[0][1])


@app.route('/submit_dd', methods=['GET', 'POST'])
def submit_dd():
    if request.method == 'POST':
        dep1 = int(request.form['dep1'])
        dep2 = int(request.form['dep2'])
        dep3 = int(request.form['dep3'])
        dep4 = int(request.form['dep4'])
        dep5 = int(request.form['dep5'])

        dep_score = dep1 + dep2 + dep3 + dep4 + dep5

        anx1 = int(request.form['anx1'])
        anx2 = int(request.form['anx2'])
        anx3 = int(request.form['anx3'])
        anx4 = int(request.form['anx4'])
        anx5 = int(request.form['anx5'])

        anx_score = anx1 + anx2 + anx3 + anx4 + anx5

        bplr1 = int(request.form['bplr1'])
        bplr2 = int(request.form['bplr2'])
        bplr3 = int(request.form['bplr3'])
        bplr4 = int(request.form['bplr4'])
        bplr5 = int(request.form['bplr5'])

        bplr_score = bplr1 + bplr2 + bplr3 + bplr4 + bplr5

        ptsd1 = int(request.form['ptsd1'])
        ptsd2 = int(request.form['ptsd2'])
        ptsd3 = int(request.form['ptsd3'])
        ptsd4 = int(request.form['ptsd4'])
        ptsd5 = int(request.form['ptsd5'])

        ptsd_score = ptsd1 + ptsd2 + ptsd3 + ptsd4 + ptsd5

        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        import numpy as np

        # Data
        data = [dep_score, anx_score, bplr_score, ptsd_score]

        # Colors
        colors = ["#FF4500", "#4B0082", "#00FF00", "#FFFF00"]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the horizontal bars
        bars = ax.barh(np.arange(len(data)), data, color=colors)

        # Add labels and title
        ax.set_xlabel('Value')
        ax.set_ylabel('Category')
        ax.set_title('Breakdown of Unstability Characteristics')

        # Create the vertical line
        ax.axvline(x=50, color='black', linestyle='--', linewidth=1)

        # Set the x-axis limits
        ax.set_xlim(0, np.max(data) + 20)

        # Set the y-axis limits
        ax.set_ylim(-0.5, len(data) - 0.5)

        for i, value in enumerate(data):
            # Calculate the percentage
            percentage = value / 50 * 100

            # Add the percentage to the plot
            ax.annotate(str(int(percentage)) + "%", (value, i - 0.2), color='black', ha='left')

            # Set the tick labels
            ax.set_yticks(np.arange(len(data)))
            ax.set_yticklabels(['DEPRESSION', 'ANXIETY', 'BIPOLAR', 'PTSD'])

        # Show the plot
        plt.tight_layout()
        plt.savefig(r'C:\Users\Dell\Desktop\PSYCHOLOGICAL INSTABILITY\static\uploads\bar_plot.png')
        plt.close()

        theta = np.linspace(0, 2 * np.pi, 360)
        radii = np.array([dep_score, anx_score, bplr_score, ptsd_score])
        # radii = np.array([10,20,30,40])

        colors = ['red', 'green', 'blue', 'yellow', 'black']

        ax2 = plt.subplot(111, projection='polar')

        for i in range(len(radii)):
            # Plot the curve
            ax2.plot(theta[i * 90: (i + 1) * 90], radii[i] * np.ones_like(theta[i * 90: (i + 1) * 90]), color=colors[i])

            # Shade the region under the curve
            ax2.fill_between(theta[i * 90: (i + 1) * 90], 0, radii[i] * np.ones_like(theta[i * 90: (i + 1) * 90]),
                             color=colors[i], alpha=0.3)

        ax2.set_rlim(0, np.max(radii))

        plt.savefig(r'C:\Users\Dell\Desktop\PSYCHOLOGICAL INSTABILITY\static\uploads\polar_plot.png')

        return render_template('unstable.html')


if __name__ == '__main__':
    app.run(debug=True)
