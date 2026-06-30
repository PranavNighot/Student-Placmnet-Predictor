import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# Configures the page title and icon
st.set_page_config(page_title="Student Placement Analysis", page_icon="🎓", layout="wide")

# Title and Informative Header
st.title("🎓 Student Placement Prediction & Analysis")
st.markdown("---")
def main_page():
    st.markdown("Home Page")
   
  

 # Informative Image (using a public URL for a professional look)
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&q=80&w=1000", 
         caption="Empowering Students for a Brighter Future", use_container_width=True)

    st.write("""
 ##  Project Overview
 The Student Placement dataset contains information about students’ academics, technical skills, soft skills, and their project or internship experience. 
 By analyzing this data, we can understand overall student performance and identify the key factors that influence placement results.
 """)

    st.info("👈 Use the sidebar to navigate through the 5 pages of this project!")

def page2():
    st.title("📊 Dataset Overview")
    # EVERYTHING below must be indented to stay inside Page 2
    st.write("### Raw Data Processing")
    
    # 1. Load the data
    try:
        df = pd.read_csv("train_student.csv")
        df1 = pd.read_csv("test_student.csv")
        # 2. Define desired columns
        cleaned_columns = [
            'Student_ID', 'Age', 'Gender', 'Degree', 'Branch', 'CGPA', 
            'Internships', 'Projects', 'Coding_Skills', 'Communication_Skills', 
            'Aptitude_Test_Score', 'Soft_Skills_Rating', 'Certifications', 'Backlogs'
        ]

        # 3. Apply cleaning
        df_cleaned = df[cleaned_columns + ['Placement_Status']].copy()
        df1_cleaned = df1[cleaned_columns].copy()

        # 4. Display ONLY on this page
        st.subheader("Cleaned Train Data")
        st.dataframe(df_cleaned.head()) # .dataframe looks better than .write

        st.subheader("Cleaned Test Data")
        st.dataframe(df1_cleaned.head())

        # 5. Store in session state for use in other pages (like Predictor)
        st.session_state['train_data'] = df_cleaned
        st.session_state['test_data'] = df1_cleaned
    except FileNotFoundError:
        st.error("CSV files not found. Please ensure 'train_student.csv' and 'test_student.csv' are in the project folder.")
        st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&q=80&w=1000", width=700)

    st.write("#### The Student Placement dataset contains information about students’ academics, technical skills, soft skills, and their project or internship experience. By analyzing this data, we can understand overall student performance, identify the key factors that influence placement results, and find areas where students may need improvement. This analysis helps colleges and training teams understand what skills matter most for getting placed and supports better planning to improve student employability and placement outcomes.")



def page3():
    st.title("📈 Visualizations")

    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

    st.write("### Placement Distribution")
   # Line explanation: Creating a figure for Matplotlib to display in Streamlit.
    
    if 'train_data' in st.session_state:
        train = st.session_state['train_data']

        st.subheader("📊 1.Distribution of Placement Status")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(
            x='Placement_Status',
            data=train,
            palette='viridis',
            ax=ax
        )

        ax.set_title('Distribution of Placement Status')
        ax.set_xlabel('Not Placed vs Placed')
        ax.set_ylabel('Number of Students')

        st.pyplot(fig)
    else:
        st.warning("Please open the Data Overview page first to load the dataset.")
    st.write("### It tells us if the data is balanced. If one bar is much taller than the other, our model might become biased toward that outcome.")
     
    if 'train_data' in st.session_state:
        train = st.session_state['train_data']
        
        st.subheader("📊 2.CGPA Distribution by Placement Status")
     
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            x='Placement_Status',
            y='CGPA',
            data=train,
            palette='Set2',
            ax=ax2
        )

        ax2.set_title('CGPA Distribution by Placement Status')
        ax2.set_xlabel('Placement Status')
        ax2.set_ylabel('CGPA')

        st.pyplot(fig2)

    else:
        st.warning("⚠️ Please open the 'Data Overview' page first to load the dataset.")
    st.write("### The box plot shows the middle range of grades. If the Placed box is higher up on the CGPA axis than the Not Placed box, it proves that students with higher CGPAs are more likely to get jobs.")

    if 'train_data' in st.session_state:
        train = st.session_state['train_data']
        
        st.subheader("📊 3.Impact of Internships on Placement")

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.countplot(
        x='Internships',
        hue='Placement_Status',
        data=train,
        palette='Set1',
        ax=ax3
        )

        ax3.set_title('Impact of Internships on Placement')
        ax3.set_xlabel('Number of Internships')
        ax3.set_ylabel('Number of Students')
        ax3.legend(title='Placement Status')

        st.pyplot(fig3)
    st.write("### This chart compares students with 0, 1, or 2+ internships. You will likely notice that as the number of internships increases, the orange bar (Placed) becomes much taller than the blue bar (Not Placed), showing that experience matters.")
   
    if 'train_data' in st.session_state:
        train = st.session_state['train_data']
        
        st.subheader("📊 4.Placement Status Across Different Branch")
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        sns.countplot(
        y='Branch',
        hue='Placement_Status',
        data=train,
        palette='viridis',
        ax=ax4)

        ax4.set_title('Placement Status Across Different Branches')
        ax4.set_xlabel('Number of Students')
        ax4.set_ylabel('Branch')
        ax4.legend(title='Placement Status')

        st.pyplot(fig4)
    st.write("### This chart lists the departments on the side. It helps us identify if certain fields (like CSE or IT) have an easier time getting placed compared to others.")
    
    
    if 'train_data' in st.session_state:
        train = st.session_state['train_data']    
        st.subheader("📊 5.Correlation Heatmap of Features")
        # Calculate correlation matrix for numerical columns
        
        correlation = train.select_dtypes(include=['int64', 'float64']).corr()

        # Create figure
        fig5, ax5 = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        ax=ax5
        )

    # Set title
    ax5.set_title('Correlation Heatmap of Features')

    # Display in Streamlit
    st.pyplot(fig5)
    


def page4():
    st.title("🤖 Placement Predictor")
    st.markdown("### 🎯 Check Your Placement Chances Instantly")

  # Load model
    @st.cache_resource
    def load_model():
     with open("model.pkl", "rb") as file:
        model = pickle.load(file)
     return model

    model = load_model()

    st.markdown("---")

    # 🎨 Use columns for better layout
    col1, col2 = st.columns(2)

    with col1:
     st.subheader("📘 Academic Details")
     age = st.number_input("Age", 18, 30, 21)
     cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
     backlogs = st.slider("Backlogs", 0, 10, 0)

     st.subheader("💻 Technical Skills")
     coding = st.slider("Coding Skills", 1, 10, 5)
     projects = st.slider("Projects", 0, 10, 2)
     internships = st.slider("Internships", 0, 5, 1)

    with col2:
     st.subheader("🧠 Aptitude & Soft Skills")
     aptitude = st.slider("Aptitude Score", 0, 100, 60)
     communication = st.slider("Communication Skills", 1, 10, 5)
     soft_skills = st.slider("Soft Skills", 1, 10, 5)

     st.subheader("📜 Additional Info")
     certifications = st.slider("Certifications", 0, 10, 1)
     gender = st.selectbox("Gender", ["Male", "Female"])
     degree = st.selectbox("Degree", ["B.Tech", "B.Sc", "B.Com"])
     branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "Mechanical"])

    st.markdown("---")

    # Encoding (unchanged)
    gender = 1 if gender == "Male" else 0
    degree_map = {"B.Tech": 0, "B.Sc": 1, "B.Com": 2}
    branch_map = {"CSE": 0, "IT": 1, "ECE": 2, "Mechanical": 3}

    degree = degree_map[degree]
    branch = branch_map[branch]

    # Input DataFrame (unchanged)
    input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Degree': [degree],
    'Branch': [branch],
    'CGPA': [cgpa],
    'Internships': [internships],
    'Projects': [projects],
    'Coding_Skills': [coding],
    'Communication_Skills': [communication],
    'Aptitude_Test_Score': [aptitude],
    'Soft_Skills_Rating': [soft_skills],
    'Certifications': [certifications],
    'Backlogs': [backlogs]
})

     # 🎯 Centered button
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Placement")
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction (unchanged)
    if predict_btn:
     prediction = model.predict(input_data)[0]

     st.markdown("---")

     if prediction == 1:
        st.success("🎉 Congratulations! You are likely to be PLACED")
        st.snow()
     else:
      st.error("❌ You may NOT be placed. Keep improving your skills!")
def page5():
    import streamlit as st

    st.title("📌 About This Project")
    st.markdown("---")

    # 🌟 Intro Image
    st.image(
        "https://images.unsplash.com/photo-1553877522-43269d4ea984",
        caption="Data-driven decisions for student success",
        use_container_width=True
    )

    # 📖 Project Description
    st.header("🎓 Project Overview")
    st.write("""
    This project is built to **analyze student data and predict placement chances**.

    It uses Machine Learning to understand how different factors like:
    - Academic performance (CGPA)
    - Skills (Coding, Communication)
    - Experience (Internships, Projects)
    - Certifications & Soft Skills

    affect whether a student gets placed or not.
    """)

    # 📊 Workflow Image
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        caption="From Data → Analysis → Prediction",
        use_container_width=True
    )

    # ⚙️ How It Works
    st.header("⚙️ How This Project Works")
    st.write("""
    1️⃣ Data is collected from student records  
    2️⃣ Data is cleaned and prepared  
    3️⃣ Machine Learning model is trained  
    4️⃣ Model learns patterns from data  
    5️⃣ User enters student details  
    6️⃣ Model predicts placement result  

    👉 Output: **Placed or Not Placed**
    """)

    # 🧠 ML Section
    st.header("🤖 Machine Learning Model")
    st.write("""
    We used a Machine Learning model to make predictions.

    ✔ Handles both numerical and categorical data  
    ✔ Uses preprocessing techniques like encoding  
    ✔ Gives fast and accurate predictions  

    The model is saved using **Pickle (.pkl)** and used in this app.
    """)

    # 🛠️ Tools Used
    st.header("🛠️ Tools & Technologies")
    st.write("""
    - Python 🐍  
    - Pandas & NumPy (Data Handling)  
    - Matplotlib & Seaborn (Visualization)  
    - Scikit-learn (Machine Learning)  
    - Streamlit (Web App)  
    """)

    # 🎯 Objective
    st.header("🎯 Project Objective")
    st.write("""
    The main goal of this project is to:

    ✔ Help students understand their placement chances  
    ✔ Identify important factors for placement  
    ✔ Provide a simple and interactive dashboard  
    ✔ Apply Machine Learning in real-world scenario  
    """)

    # 📷 Final Image
    st.image(
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f",
        caption="Empowering students with insights",
        use_container_width=True
    )

    # 🙌 Footer
    st.markdown("---")
    st.success("✨ This project demonstrates how data and AI can help students grow and succeed.")
 
page_names_to_func={
  "Home Page":main_page,
    "Data Overview":page2,
    "Visualizations":page3,
    "Placement Predictor":page4,
    "About":page5
    }

selected_page=st.sidebar.selectbox("Select a Page",page_names_to_func.keys())
page_names_to_func[selected_page]()
