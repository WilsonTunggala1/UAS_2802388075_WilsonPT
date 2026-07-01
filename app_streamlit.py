import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

# Gunakan cache agar model tidak di-load ulang setiap kali user mengubah input
@st.cache_resource
def load_model():
    # Pastikan path ini sesuai dengan tempat Anda menyimpan model terbaik
    return joblib.load('artifacts/best_credit_score_pipeline.pkl')

model = load_model()

def main():
    st.title('💳 Intelligent Credit Score Predictor')
    st.markdown("Masukkan informasi finansial nasabah di bawah ini untuk memprediksi kategori *Credit Score* mereka.")

    # Membuat Tabs/Kolom untuk mengelompokkan input agar UI tidak terlalu panjang ke bawah
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("👤 Profil Personal")
        month = st.selectbox("Month of Record", ["January", "February", "March", "April", "May", "June", "July", "August"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", [
            "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
            "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", 
            "Accountant", "Musician", "Mechanic", "Writer", "Architecture", "Other"
        ])
        
        st.header("💰 Informasi Pendapatan")
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=(annual_income/12), step=100.0)
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=500.0, step=100.0)
        amount_invested = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=100.0, step=50.0)

    with col2:
        st.header("🏦 Rekening & Kartu Kredit")
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=3)
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=15, value=2)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=15.0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=20, value=2)
        credit_history_months = st.number_input("Credit History Age (in Months)", min_value=0.0, value=150.0)
        
        st.header("⚠️ Riwayat Keterlambatan")
        delay_due_date = st.number_input("Average Delay from Due Date (Days)", min_value=0, value=14)
        num_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=4)

    with col3:
        st.header("📉 Rasio & Perilaku Utang")
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1500.0, step=100.0)
        utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
        total_emi = st.number_input("Total EMI per Month ($)", min_value=0.0, value=200.0, step=50.0)
        changed_limit = st.number_input("Changed Credit Limit (%)", value=10.0)
        
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        min_amount = st.selectbox("Payment of Minimum Amount", ["No", "NM", "Yes"])
        payment_behavior = st.selectbox("Payment Behaviour", [
            "Low_spent_Small_value_payments", "High_spent_Medium_value_payments", 
            "Low_spent_Medium_value_payments", "High_spent_Large_value_payments", 
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments", "Other"
        ])

    st.markdown("---")
    st.header("📝 Portofolio Pinjaman (Loans)")
    # UI yang elegan: Multiselect untuk mempermudah user
    loan_types = ["Auto_Loan", "Credit_Builder_Loan", "Debt_Consolidation_Loan", 
                  "Home_Equity_Loan", "Mortgage_Loan", "Not_Specified", "Payday_Loan", 
                  "Personal_Loan", "Student_Loan", "Unknown"]
    
    selected_loans = st.multiselect("Pilih semua jenis pinjaman yang dimiliki nasabah:", loan_types)

    # Dictionary komprehensif untuk disatukan menjadi DataFrame
    if st.button("🔍 Make Prediction", type="primary"):
        # 1. Konversi multiselect menjadi biner (0/1) untuk masing-masing kolom loan
        loan_dict = {loan: (1 if loan in selected_loans else 0) for loan in loan_types}

        model_occupation = "Unknown" if occupation == "Other" else occupation
        model_payment_behavior = "Unknown" if payment_behavior == "Other" else payment_behavior

        # 2. Gabungkan data utama dengan data loan
        data_input = {
            'Month': month,
            'Age': age,
            'Occupation': model_occupation,
            'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_salary,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': len(selected_loans), # Otomatis dihitung berdasarkan multiselect
            'Delay_from_due_date': delay_due_date,
            'Num_of_Delayed_Payment': num_delayed_payment,
            'Changed_Credit_Limit': changed_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Credit_Mix': credit_mix,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': utilization_ratio,
            'Payment_of_Min_Amount': min_amount,
            'Total_EMI_per_month': total_emi,
            'Amount_invested_monthly': amount_invested,
            'Payment_Behaviour': model_payment_behavior,
            'Monthly_Balance': monthly_balance,
            'Credit_History_Age_Months': credit_history_months
        }
        
        # Merge dictionary
        data_input.update(loan_dict)
        
        # 3. Buat DataFrame
        # Pastikan kolom sesuai dengan urutan dan penamaan yang diharapkan preprocessor
        df = pd.DataFrame([data_input])

        # 4. Lakukan Prediksi
        try:
            # Model akan memproses Pipeline (Preprocessing -> Prediksi)
            prediction_encoded = model.predict(df)[0]
            
            # Mapping ulang output angka (0, 1, 2) kembali ke String yang mudah dibaca
            score_mapping = {0: "Poor", 1: "Standard", 2: "Good"}
            final_prediction = score_mapping[prediction_encoded]
            
            # Tampilkan hasil dengan warna yang sesuai (Hijau/Kuning/Merah)
            st.markdown("### Hasil Prediksi:")
            if final_prediction == "Good":
                st.success(f"🌟 Credit Score: **{final_prediction}** - Nasabah berisiko rendah.")
            elif final_prediction == "Standard":
                st.warning(f"⚖️ Credit Score: **{final_prediction}** - Nasabah memiliki risiko menengah.")
            else:
                st.error(f"⚠️ Credit Score: **{final_prediction}** - Nasabah berisiko tinggi (Ditolak).")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada pemrosesan: {e}")

if __name__ == "__main__":
    main()