from flask import Flask, render_template
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# Load the metrics DataFrame
metrics_df = pd.read_csv('../Output_files/metrics.csv')

@app.route('/')
def dashboard():
    # Create a bar plot for accuracy
    fig = px.bar(metrics_df, x='Period', y='Accuracy', title='Model Accuracy Over Periods')

    # Convert plotly figure to HTML
    plot_html = fig.to_html(full_html=False)

    return render_template('index.html', plot_html=plot_html)


if __name__ == '__main__':
    app.run(debug=True)
