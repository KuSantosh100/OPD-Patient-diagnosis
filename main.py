import main as st
import pandas as pd
import joblib
import numpy as np

# Load the trained Random Forest model and feature columns
model = joblib.load('random_forest_model.pkl')
feature_columns = joblib.load('feature_columns_.pkl')

# Define the standardized categories
SEX_OPTIONS = ['M', 'F']
SOCIOECONOMIC_STATUS_OPTIONS = ['Upper Class', 'Lower Class', 'Upper Middle Class', 'Lower Middle Class']

# Load the original dataset to get unique categorical values for other dropdowns
df_original = pd.read_csv("./data/OPMD-PATIENTS.csv")
habbits_options = df_original['HABBITS'].dropna().unique().tolist()
duration_options = df_original['DURATION'].dropna().unique().tolist()

# Streamlit App Layout
st.title('🏥 Patient Diagnosis Prediction System')
st.markdown('---')
st.write('Enter patient details below to get a diagnosis prediction using our trained Random Forest model.')

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('Patient Demographics')
    age = st.number_input('Age', min_value=1, max_value=100, value=30, help='Enter patient age in years')
    sex = st.selectbox('Sex', options=SEX_OPTIONS, help='Select patient gender')
    socioeconomic_status = st.selectbox(
        'Socioeconomic Status', 
        options=SOCIOECONOMIC_STATUS_OPTIONS, 
        help='Select patient socioeconomic background'
    )

with col2:
    st.subheader('Lifestyle & Medical History')
    habbits = st.selectbox('Habits', options=habbits_options, help='Select patient habits')
    duration = st.selectbox('Duration', options=duration_options, help='Select duration of habits/symptoms')

def preprocess_input(age, sex, socioeconomic_status, habbits, duration):
    """
    Preprocess the input data to match the exact format expected by the trained model.
    """
    # Create a DataFrame with zeros for all feature columns
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Set the age
    input_df['AGE'] = age
    
    # Set the appropriate sex column to 1
    sex_column = f'SEX_{sex}'
    if sex_column in input_df.columns:
        input_df[sex_column] = 1
    
    # Set the appropriate socioeconomic status column to 1
    socio_column = f'SOCIOECONOMIC STATUS_{socioeconomic_status}'
    if socio_column in input_df.columns:
        input_df[socio_column] = 1
    
    # Set the appropriate habbits column to 1
    habbits_column = f'HABBITS_{habbits}'
    if habbits_column in input_df.columns:
        input_df[habbits_column] = 1
    
    # Set the appropriate duration column to 1
    duration_column = f'DURATION_{duration}'
    if duration_column in input_df.columns:
        input_df[duration_column] = 1
    
    return input_df

# Prediction section
st.markdown('---')
st.subheader('🔬 Diagnosis Prediction')

if st.button('🚀 Predict Diagnosis', type='primary'):
    try:
        # Preprocess the input data
        input_data_processed = preprocess_input(age, sex, socioeconomic_status, habbits, duration)
        
        # Make prediction
        prediction = model.predict(input_data_processed)
        prediction_proba = model.predict_proba(input_data_processed)
        
        # Display results
        st.success(f'🎯 **Predicted Diagnosis: {prediction[0].replace("_", " ").title()}**')
        
        # Display prediction probabilities
        st.subheader('📊 Prediction Confidence')
        classes = model.classes_
        probabilities = prediction_proba[0]
        
        prob_df = pd.DataFrame({
            'Diagnosis': [cls.replace("_", " ").title() for cls in classes],
            'Probability': probabilities,
            'Confidence %': [f'{prob*100:.1f}%' for prob in probabilities]
        }).sort_values('Probability', ascending=False)
        
        # Create a bar chart for probabilities
        st.bar_chart(prob_df.set_index('Diagnosis')['Probability'])
        
        # Display the probability table
        st.dataframe(prob_df, use_container_width=True)
        
        # Add interpretation
        max_prob = probabilities.max()
        if max_prob > 0.7:
            confidence_level = "High"
            confidence_color = "🟢"
        elif max_prob > 0.5:
            confidence_level = "Medium"
            confidence_color = "🟡"
        else:
            confidence_level = "Low"
            confidence_color = "🔴"
        
        st.info(f'{confidence_color} **Model Confidence: {confidence_level}** ({max_prob*100:.1f}%)')
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.write("Please check your input values and try again.")

# Information section
st.markdown('---')
st.subheader('ℹ️ Model Information')

with st.expander("View Model Details"):
    st.write("**Model Type:** Random Forest Classifier")
    st.write("**Training Accuracy:** ~78.4%")
    st.write("**Number of Features:**", len(feature_columns))
    st.write("**Possible Diagnoses:**")
    diagnoses = [
        "Oral Submucous Fibrosis",
        "Oral Lichen Planus", 
        "Leukoplakia",
        "Erythroplakia",
        "Frictional Keratosis",
        "Tobacco Pouch Keratosis",
        "Smoker's Palate",
        "Other"
    ]
    for diagnosis in diagnoses:
        st.write(f"• {diagnosis}")

# Feature importance section
with st.expander("View Feature Importance"):
    st.write("**Top 10 Most Important Features for Prediction:**")
    st.write("1. Age (27.0%)")
    st.write("2. Habits - Paan (4.1%)")
    st.write("3. Socioeconomic Status - Upper Middle Class (3.9%)")
    st.write("4. Duration - 1 year (3.8%)")
    st.write("5. Habits - Cigarette (3.8%)")
    st.write("6. Sex - Female (3.8%)")
    st.write("7. Duration - 2 years (3.4%)")
    st.write("8. Socioeconomic Status - Lower Middle Class (3.4%)")
    st.write("9. Socioeconomic Status - Lower Class (3.3%)")
    st.write("10. Habits - Bidi & Alcohol (3.2%)")

# Debug section (hidden by default)
if st.checkbox("🔧 Show Debug Information"):
    st.write("**Expected feature columns:**")
    st.write(feature_columns)
    
    if st.button("Test Preprocessing"):
        test_input = preprocess_input(age, sex, socioeconomic_status, habbits, duration)
        st.write("**Preprocessed input shape:**", test_input.shape)
        st.write("**Non-zero features:**")
        non_zero_features = test_input.loc[0, test_input.loc[0] != 0]
        st.write(non_zero_features)

# Footer
st.markdown('---')
st.markdown('*This application is for educational and research purposes. Please consult with medical professionals for actual diagnosis and treatment.*')