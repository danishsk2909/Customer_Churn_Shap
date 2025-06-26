import numpy as np
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Load trained RandomForest model
model = pickle.load(open("RFC_Model.pkl", "rb"))

# Define features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
                        'PaymentMethod', 'DeviceProtection', 'OnlineBackup',
                        'StreamingMovies', 'StreamingTV']
feature_names = numerical_features + categorical_features

# Load and preprocess training data
X_train_raw = pd.read_csv("Telco-Customer-Churn.csv")
X_train_raw['TotalCharges'] = pd.to_numeric(X_train_raw['TotalCharges'], errors='coerce')

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    le.fit(X_train_raw[col].astype(str))
    label_encoders[col] = le

X_train_processed = X_train_raw[feature_names].copy()
X_train_processed[numerical_features] = X_train_processed[numerical_features].apply(pd.to_numeric, errors='coerce')
X_train_processed.dropna(inplace=True)

for col in categorical_features:
    X_train_processed[col] = label_encoders[col].transform(X_train_processed[col].astype(str))

# Rule-based logic
def rule_based_risk(form_data):
    try:
        high_risk_conditions = [
            form_data['Contract'] == 'Month-to-month',
            form_data['TechSupport'] == 'No',
            form_data['OnlineSecurity'] == 'No',
            form_data['InternetService'] == 'Fiber optic',
            form_data['PaymentMethod'] == 'Electronic check',
            form_data['DeviceProtection'] == 'No',
            form_data['OnlineBackup'] == 'No',
            form_data['StreamingMovies'] == 'Yes',
            form_data['StreamingTV'] == 'Yes',
            float(form_data['tenure']) < 6,
            float(form_data['MonthlyCharges']) > 80,
            float(form_data['TotalCharges']) < 200,
        ]

        medium_risk_conditions = [
            form_data['Contract'] == 'One year',
            form_data['TechSupport'] == 'No',
            form_data['OnlineSecurity'] == 'No',
            form_data['InternetService'] == 'DSL',
            form_data['DeviceProtection'] == 'No',
            6 <= float(form_data['tenure']) < 12,
            60 <= float(form_data['MonthlyCharges']) <= 80,
            200 <= float(form_data['TotalCharges']) < 500,
        ]

        high_score = sum(high_risk_conditions)
        medium_score = sum(medium_risk_conditions)

        if high_score >= 6:
            return "High"
        elif medium_score >= 4:
            return "Medium"
        else:
            return "Low"
    except Exception as e:
        print(f"Rule-based risk calculation error: {e}")
        return "Unknown"

def create_shap_force_plot(shap_values, base_value, feature_names, feature_values, output_value):
    """Create a SHAP force plot similar to the one in your image"""
    
    try:
        # Ensure all inputs are proper numpy arrays/scalars
        shap_values = np.array(shap_values).flatten()
        feature_values = np.array(feature_values).flatten()
        base_value = float(base_value)
        output_value = float(output_value)
        
        print(f"Debug - Input shapes: shap={shap_values.shape}, features={feature_values.shape}")
        print(f"Debug - Base value: {base_value}, Output value: {output_value}")
        
        # Ensure we have the same number of features and SHAP values
        min_len = min(len(shap_values), len(feature_values), len(feature_names))
        shap_values = shap_values[:min_len]
        feature_values = feature_values[:min_len]
        feature_names = feature_names[:min_len]
        
        # Sort features by absolute SHAP value
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values,
            'feature_value': feature_values
        })
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        shap_df = shap_df.sort_values('abs_shap', ascending=False)
        
        # Take top features for visualization
        top_features = shap_df.head(8)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Set up the plot
        ax.set_xlim(-0.15, 0.65)
        ax.set_ylim(-0.5, 2.5)
        
        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add scale at top
        scale_y = 2.2
        ax.text(-0.1, scale_y + 0.1, '-0.1', ha='center', va='bottom', fontsize=10)
        ax.text(0.0, scale_y + 0.1, '0.0', ha='center', va='bottom', fontsize=10)
        ax.text(0.1, scale_y + 0.1, '0.1', ha='center', va='bottom', fontsize=10)
        ax.text(0.2, scale_y + 0.1, '0.2', ha='center', va='bottom', fontsize=10)
        ax.text(0.3, scale_y + 0.1, '0.3', ha='center', va='bottom', fontsize=10)
        ax.text(0.4, scale_y + 0.1, '0.4', ha='center', va='bottom', fontsize=10)
        ax.text(0.5, scale_y + 0.1, '0.5', ha='center', va='bottom', fontsize=10)
        ax.text(0.6, scale_y + 0.1, '0.6', ha='center', va='bottom', fontsize=10)
        
        # Add scale line
        ax.plot([-0.1, 0.6], [scale_y, scale_y], 'k-', linewidth=1)
        for x in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            ax.plot([x, x], [scale_y - 0.02, scale_y + 0.02], 'k-', linewidth=1)
        
        # Add legend
        ax.text(0.15, scale_y + 0.2, 'higher', ha='center', va='center', fontsize=12, color='#FF1744', weight='bold')
        ax.text(0.35, scale_y + 0.2, 'lower', ha='center', va='center', fontsize=12, color='#2196F3', weight='bold')
        
        # Add arrow
        ax.annotate('', xy=(0.32, scale_y + 0.2), xytext=(0.18, scale_y + 0.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        ax.text(0.25, scale_y + 0.25, 'output value', ha='center', va='bottom', fontsize=10, color='gray')
        ax.text(0.5, scale_y + 0.25, 'base value', ha='center', va='bottom', fontsize=10, color='gray')
        
        # Create the main force plot bar
        bar_height = 0.3
        bar_y = 1.0
        
        # Start from base value
        current_x = base_value
        
        # Draw segments for each feature
        positive_features = top_features[top_features['shap_value'] > 0].copy()
        negative_features = top_features[top_features['shap_value'] < 0].copy()
        
        if not positive_features.empty:
            positive_features = positive_features.sort_values('shap_value', ascending=False)
        if not negative_features.empty:
            negative_features = negative_features.sort_values('shap_value', ascending=True)
        
        # Draw positive contributions (red/pink)
        total_features = len(positive_features) + len(negative_features)
        segment_height = bar_height / max(1, total_features) * 4
        y_offset = 0
        
        for _, row in positive_features.iterrows():
            width = float(row['shap_value'])  # Ensure scalar
            if width > 0:  # Safety check
                # Create gradient effect with multiple thin rectangles
                for i in range(10):
                    alpha = 0.3 + 0.7 * (i / 9)  # Gradient alpha
                    rect_width = width / 10
                    rect = FancyBboxPatch((current_x + i * rect_width, bar_y + y_offset), 
                                        rect_width, segment_height,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#FF1744', alpha=alpha, edgecolor='none')
                    ax.add_patch(rect)
                
                # Add chevron pattern
                chevron_count = max(1, int(width * 100))
                for i in range(0, chevron_count, 4):  # Every 4th position
                    x_pos = current_x + (i / 100.0) * width
                    if x_pos < current_x + width:  # Stay within bounds
                        ax.plot([x_pos, x_pos + 0.005], [bar_y + y_offset + 0.05, bar_y + y_offset + 0.15], 
                               'white', alpha=0.7, linewidth=1)
                        ax.plot([x_pos + 0.005, x_pos + 0.01], [bar_y + y_offset + 0.15, bar_y + y_offset + 0.05], 
                               'white', alpha=0.7, linewidth=1)
                
                current_x += width
                y_offset += segment_height * 0.1
        
        # Draw negative contributions (blue)
        for _, row in negative_features.iterrows():
            width = abs(float(row['shap_value']))  # Ensure positive scalar
            if width > 0:  # Safety check
                start_x = current_x - width
                
                # Create gradient effect
                for i in range(10):
                    alpha = 0.3 + 0.7 * (i / 9)
                    rect_width = width / 10
                    rect = FancyBboxPatch((start_x + i * rect_width, bar_y + y_offset), 
                                        rect_width, segment_height,
                                        boxstyle="round,pad=0.01",
                                        facecolor='#2196F3', alpha=alpha, edgecolor='none')
                    ax.add_patch(rect)
                
                # Add chevron pattern
                chevron_count = max(1, int(width * 100))
                for i in range(0, chevron_count, 4):
                    x_pos = start_x + (i / 100.0) * width
                    if x_pos < start_x + width:  # Stay within bounds
                        ax.plot([x_pos, x_pos + 0.005], [bar_y + y_offset + 0.05, bar_y + y_offset + 0.15], 
                               'white', alpha=0.7, linewidth=1)
                        ax.plot([x_pos + 0.005, x_pos + 0.01], [bar_y + y_offset + 0.15, bar_y + y_offset + 0.05], 
                               'white', alpha=0.7, linewidth=1)
                
                current_x = start_x
                y_offset += segment_height * 0.1
        
        # Add output value marker
        ax.plot([output_value, output_value], [bar_y - 0.1, bar_y + bar_height + 0.1], 'k-', linewidth=2)
        ax.text(output_value, bar_y + bar_height + 0.15, f'{output_value:.3f}', ha='center', va='bottom', 
                fontsize=12, weight='bold')
        
        # Add base value marker
        ax.plot([base_value, base_value], [bar_y - 0.1, bar_y + bar_height + 0.1], 'k--', linewidth=1, alpha=0.7)
        
        # Add feature labels below
        label_y = 0.3
        
        # Position labels
        x_positions = []
        current_x = base_value
        
        for _, row in positive_features.iterrows():
            width = float(row['shap_value'])
            if width > 0:
                x_positions.append((row['feature'], current_x + width/2, row['shap_value'], row['feature_value']))
                current_x += width
        
        temp_x = current_x
        for _, row in negative_features.iterrows():
            width = abs(float(row['shap_value']))
            if width > 0:
                temp_x -= width
                x_positions.append((row['feature'], temp_x + width/2, row['shap_value'], row['feature_value']))
        
        # Add feature labels
        for feature_name, x_pos, shap_val, feature_val in x_positions:
            # Create rounded rectangle background
            if float(shap_val) > 0:
                bg_color = '#FFE8E8'
                text_color = '#FF1744'
            else:
                bg_color = '#E8F4FD'
                text_color = '#2196F3'
            
            # Add background rectangle
            bbox = dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.8, edgecolor='none')
            ax.text(x_pos, label_y, f'{feature_name}', ha='center', va='center', 
                    fontsize=10, weight='bold', color=text_color, bbox=bbox)
        
        plt.tight_layout()
        
        # Save to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64
        
    except Exception as e:
        print(f"Error in create_shap_force_plot: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a simple fallback plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'SHAP Force Plot Error:\n{str(e)}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffcccc'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---------- 1. Input preprocessing ----------
        input_data = []
        for feature in feature_names:
            value = request.form[feature]
            if feature in numerical_features:
                input_data.append(float(value))
            else:
                encoder = label_encoders[feature]
                if value in encoder.classes_:
                    input_data.append(encoder.transform([value])[0])
                else:
                    return f"Invalid value '{value}' for feature '{feature}'"

        input_df = pd.DataFrame([input_data], columns=feature_names)

        # ---------- 2. Model prediction ----------
        prediction_proba = model.predict_proba(input_df)
        risk_score = prediction_proba[0][1]  # Probability of churn
        risk_category = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.3 else "Low"

        if risk_score > 0.7:
            retention_action = ['Grant loyalty benefits', 'Offer cashback offers', 'Schedule agent call to customer']
        elif risk_score > 0.3:
            retention_action = ['Grant loyalty points']
        else:
            retention_action = ['No Action Required']

        # ---------- 3. Rule-based category ----------
        rule_based_category = rule_based_risk(request.form)

        # ---------- 4. LIME Explanation ----------
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(X_train_processed),
                feature_names=feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification'
            )

            lime_explanation = lime_explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=4
            )

            lime_html = lime_explanation.as_html()
        except Exception as e:
            print(f"LIME explanation error: {e}")
            lime_html = f"<p>LIME explanation unavailable: {str(e)}</p>"

        # ---------- 5. SHAP Force Plot ----------
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            print(f"Debug - SHAP values type: {type(shap_values)}")
            print(f"Debug - SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'Not an array'}")
            print(f"Debug - Expected value type: {type(explainer.expected_value)}")
            print(f"Debug - Expected value: {explainer.expected_value}")

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                print(f"Debug - List format, length: {len(shap_values)}")
                # For binary classification, use the positive class (index 1)
                if len(shap_values) > 1:
                    shap_val = shap_values[1][0]  # Positive class, first sample
                    if isinstance(explainer.expected_value, (list, np.ndarray)):
                        base_val = explainer.expected_value[1]
                    else:
                        base_val = explainer.expected_value
                else:
                    shap_val = shap_values[0][0]  # Single class, first sample
                    base_val = explainer.expected_value
            else:
                print(f"Debug - Array format, shape: {shap_values.shape}")
                shap_val = shap_values[0]  # First sample
                base_val = explainer.expected_value

            # Convert to numpy array and ensure 1D
            shap_val = np.array(shap_val)
            if shap_val.ndim > 1:
                shap_val = shap_val.flatten()
            
            print(f"Debug - Final shap_val shape: {shap_val.shape}")
            print(f"Debug - Final shap_val: {shap_val}")

            # Handle base value conversion more carefully
            if isinstance(base_val, (list, np.ndarray)):
                if len(base_val) > 0:
                    base_val = float(base_val[0]) if hasattr(base_val[0], '__float__') else 0.5
                else:
                    base_val = 0.5
            else:
                try:
                    base_val = float(base_val)
                except (TypeError, ValueError):
                    base_val = 0.5

            print(f"Debug - Final base_val: {base_val}")

            # Get feature values for the instance
            feature_values = input_df.iloc[0].values
            print(f"Debug - Feature values shape: {feature_values.shape}")

            # Ensure feature_values is 1D
            if feature_values.ndim > 1:
                feature_values = feature_values.flatten()

            # Create the force plot
            force_plot_b64 = create_shap_force_plot(
                shap_val, base_val, feature_names, feature_values, risk_score
            )

            # Ensure static directory exists
            Path("static").mkdir(exist_ok=True)

            # Create enhanced SHAP HTML with the force plot
            shap_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .shap-container {{ max-width: 1000px; }}
                    .force-plot {{ text-align: center; margin: 20px 0; }}
                    .force-plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    .positive {{ color: #dc3545; font-weight: bold; }}
                    .negative {{ color: #007bff; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="shap-container">
                    <div class="summary">
                        <h3>SHAP Analysis Summary</h3>
                        <p><strong>Base Value (average prediction):</strong> {base_val:.4f}</p>
                        <p><strong>Final Prediction:</strong> {risk_score:.4f}</p>
                        <p><strong>Total Impact:</strong> {np.sum(shap_val):+.4f}</p>
                        <p><strong>Risk Category:</strong> {risk_category}</p>
                    </div>
                    
                    <div class="force-plot">
                        <h4>SHAP Force Plot</h4>
                        <img src="data:image/png;base64,{force_plot_b64}" alt="SHAP Force Plot">
                        <p><em>This plot shows how each feature pushes the prediction above or below the base value.</em></p>
                    </div>
                    
                    <h4>Feature Contributions</h4>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                            <th>SHAP Impact</th>
                            <th>Effect</th>
                        </tr>
            """

            # Add table rows
            feature_impacts = list(zip(feature_names, feature_values, shap_val))
            feature_impacts.sort(key=lambda x: abs(x[2]), reverse=True)

            for feature, value, impact in feature_impacts:
                impact_class = "positive" if impact > 0 else "negative"
                effect = "Increases churn risk" if impact > 0 else "Decreases churn risk"
                
                shap_html += f"""
                    <tr>
                        <td><strong>{feature}</strong></td>
                        <td>{value:.4f}</td>
                        <td class="{impact_class}">{impact:+.4f}</td>
                        <td>{effect}</td>
                    </tr>
                """

            shap_html += """
                    </table>
                </div>
            </body>
            </html>
            """

            # Save SHAP HTML
            with open("static/shap_force_plot.html", "w") as f:
                f.write(shap_html)

        except Exception as e:
            print(f"SHAP analysis error: {e}")
            # Create fallback SHAP HTML
            shap_html = f"""
            <html>
            <body>
                <div style="padding: 20px;">
                    <h3>SHAP Analysis Unavailable</h3>
                    <p>Error: {str(e)}</p>
                    <p>Risk Score: {risk_score:.4f} ({risk_category})</p>
                </div>
            </body>
            </html>
            """
            with open("static/shap_force_plot.html", "w") as f:
                f.write(shap_html)

        # ---------- 6. Return HTML ----------
        return render_template("predict.html",
                               risk_score=round(risk_score * 100, 2),
                               risk_category=risk_category,
                               retention_actions=retention_action,
                               rule_based_category=rule_based_category,
                               lime_html=lime_html)

    except Exception as e:
        print(f"General prediction error: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)