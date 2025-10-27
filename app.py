import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
from deepface import DeepFace
import shutil

st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Folder to store registered faces
TRAIN_DIR = "registered_faces"
os.makedirs(TRAIN_DIR, exist_ok=True)

ATTENDANCE_FILE = "attendance.csv"

# Initialize or fix the CSV file
def initialize_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    else:
        # Check if file is valid
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if df.empty or not all(col in df.columns for col in ["Name", "Date", "Time", "Status"]):
                df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
                df.to_csv(ATTENDANCE_FILE, index=False)
        except:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
            df.to_csv(ATTENDANCE_FILE, index=False)

initialize_csv()

# Initialize session state
if 'last_photo' not in st.session_state:
    st.session_state.last_photo = None
if 'attendance_marked' not in st.session_state:
    st.session_state.attendance_marked = False
if 'marked_user' not in st.session_state:
    st.session_state.marked_user = None
if 'marked_time' not in st.session_state:
    st.session_state.marked_time = None

# Function to normalize names (convert to lowercase for comparison)
def normalize_name(name):
    return name.strip().lower()

# Function to check if face already exists
def check_duplicate_face(new_image_path):
    registered_files = os.listdir(TRAIN_DIR)
    for file in registered_files:
        registered_path = os.path.join(TRAIN_DIR, file)
        try:
            result = DeepFace.verify(
                new_image_path,
                registered_path,
                model_name='VGG-Face',
                detector_backend='opencv',
                enforce_detection=False,
                distance_metric='cosine'
            )
            if result["verified"]:
                return os.path.splitext(file)[0]  # Return the registered name
        except:
            continue
    return None

# Function to delete user and their attendance records
def delete_user_completely(user_name, user_path):
    try:
        # Delete the user's photo
        if os.path.exists(user_path):
            os.remove(user_path)
        
        # Delete all attendance records for this user (case-insensitive)
        df = pd.read_csv(ATTENDANCE_FILE)
        if not df.empty:
            normalized_user = normalize_name(user_name)
            # Keep only records that don't match this user
            df = df[df["Name"].apply(lambda x: normalize_name(x) != normalized_user)]
            df.to_csv(ATTENDANCE_FILE, index=False)
        
        return True, "User and all attendance records deleted successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Sidebar Navigation
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio(
    "Select an option:",
    ["👤 Registration", "✅ Mark Attendance", "📊 Biometric Log History", "🗑️ Manage Users"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Instructions:**\n\n1. Register your face first\n2. Mark attendance daily\n3. View attendance history\n4. Manage registered users")

# --------------------- Page 1: Registration ---------------------
if page == "👤 Registration":
    st.title("👤 Face Registration")
    st.write("Register new users by capturing their face and name.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        name = st.text_input("Enter Full Name", placeholder="e.g., John Doe")
        st.info("💡 Make sure your face is clearly visible in good lighting.")
        st.warning("⚠️ Names are case-insensitive (John Doe = john doe)")
    
    with col2:
        photo_train = st.camera_input("📷 Capture Face for Registration")
    
    if name and photo_train:
        if st.button("✅ Register User", type="primary"):
            # Normalize the input name
            normalized_input = normalize_name(name)
            
            # Check if name already exists (case-insensitive)
            existing_files = os.listdir(TRAIN_DIR)
            name_exists = False
            existing_name = None
            
            for file in existing_files:
                existing_user = os.path.splitext(file)[0]
                if normalize_name(existing_user) == normalized_input:
                    name_exists = True
                    existing_name = existing_user
                    break
            
            if name_exists:
                st.warning(f"⚠️ User with this name already exists as '{existing_name}'!")
            else:
                try:
                    # Save temporary image
                    image = Image.open(photo_train)
                    temp_path = "temp_registration.jpg"
                    image.save(temp_path)
                    
                    # Check for duplicate face
                    duplicate_user = check_duplicate_face(temp_path)
                    
                    if duplicate_user:
                        st.error(f"❌ This face is already registered as '{duplicate_user}'!")
                        st.warning("⚠️ Same person cannot register with different names.")
                        os.remove(temp_path)
                    else:
                        # Test if face is detectable
                        test_result = DeepFace.extract_faces(temp_path, enforce_detection=False)
                        
                        if test_result:
                            # Save with original name (preserving case)
                            final_path = os.path.join(TRAIN_DIR, f"{name}.jpg")
                            shutil.move(temp_path, final_path)
                            st.success(f"✅ {name} registered successfully!")
                            st.balloons()
                        else:
                            os.remove(temp_path)
                            st.error("❌ No face detected in the image. Please try again with better lighting.")
                            
                except Exception as e:
                    st.error(f"Error during registration: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Display registered users
    st.markdown("---")
    st.subheader("📋 Registered Users")
    registered_users = [os.path.splitext(f)[0] for f in os.listdir(TRAIN_DIR)]
    if registered_users:
        st.write(f"**Total Registered Users:** {len(registered_users)}")
        cols = st.columns(4)
        for idx, user in enumerate(sorted(registered_users)):
            with cols[idx % 4]:
                st.write(f"• {user}")
    else:
        st.info("No users registered yet.")

# --------------------- Page 2: Mark Attendance ---------------------
elif page == "✅ Mark Attendance":
    st.title("✅ Mark Attendance")
    st.write("Capture your face to mark attendance for today.")
    
    # Check today's attendance status BEFORE camera input
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if df.empty or 'Date' not in df.columns:
            today_attendance = pd.DataFrame()
        else:
            today_date = datetime.now().strftime("%Y-%m-%d")
            today_attendance = df[df["Date"] == today_date]
    except Exception as e:
        today_attendance = pd.DataFrame()
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    
    # Display attendance status at the top
    st.markdown("---")
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        if st.session_state.attendance_marked and st.session_state.marked_user:
            st.success(f"✅ **ATTENDANCE MARKED**")
            st.write(f"**Name:** {st.session_state.marked_user}")
            st.write(f"**Time:** {st.session_state.marked_time}")
            st.write(f"**Status:** Present")
        else:
            st.warning("⏳ **ATTENDANCE NOT MARKED YET**")
            st.write("Please capture your photo to mark attendance")
    
    with col_status2:
        if len(today_attendance) > 0:
            st.info(f"📊 **Today's Total Attendance: {len(today_attendance)}**")
            st.write("**Marked Present:**")
            for _, row in today_attendance.iterrows():
                st.write(f"• {row['Name']} - {row['Time']}")
        else:
            st.info("📊 **No attendance marked today yet**")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("💡 Position your face clearly in the camera frame and click 'Take photo'.")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")
        st.write(f"**Date:** {current_date}")
        st.write(f"**Time:** {current_time}")
    
    with col2:
        photo_attendance = st.camera_input("📷 Capture Face for Attendance")
    
    # Check if a new photo was taken
    if photo_attendance is not None:
        photo_bytes = photo_attendance.getvalue()
        
        # Check if this is a new photo (different from last processed)
        if st.session_state.last_photo != photo_bytes:
            st.session_state.last_photo = photo_bytes
            st.session_state.attendance_marked = False
            st.session_state.marked_user = None
            
            with st.spinner("🔍 Verifying face... Please wait..."):
                df = pd.read_csv(ATTENDANCE_FILE)
                img = Image.open(photo_attendance)
                img_path = "temp_attendance.jpg"
                img.save(img_path)
                
                found = False
                matched_name = None
                
                registered_files = os.listdir(TRAIN_DIR)
                
                if not registered_files:
                    st.error("❌ No registered users found. Please register first!")
                else:
                    for file in registered_files:
                        registered_name = os.path.splitext(file)[0]
                        registered_path = os.path.join(TRAIN_DIR, file)
                        
                        try:
                            # Use less strict verification settings
                            result = DeepFace.verify(
                                img_path, 
                                registered_path,
                                model_name='VGG-Face',
                                detector_backend='opencv',
                                enforce_detection=False,
                                distance_metric='cosine'
                            )
                            
                            if result["verified"]:
                                found = True
                                matched_name = registered_name
                                now = datetime.now()
                                date_str = now.strftime("%Y-%m-%d")
                                time_str = now.strftime("%H:%M:%S")
                                
                                # Check if already marked present today (case-insensitive)
                                if df.empty:
                                    already_marked = False
                                else:
                                    already_marked = False
                                    for _, row in df.iterrows():
                                        if (normalize_name(row["Name"]) == normalize_name(registered_name) and 
                                            row["Date"] == date_str and 
                                            row["Status"] == "Present"):
                                            already_marked = True
                                            break
                                
                                if not already_marked:
                                    new_entry = pd.DataFrame(
                                        [[registered_name, date_str, time_str, "Present"]], 
                                        columns=["Name", "Date", "Time", "Status"]
                                    )
                                    df = pd.concat([df, new_entry], ignore_index=True)
                                    df.to_csv(ATTENDANCE_FILE, index=False)
                                    
                                    # Update session state
                                    st.session_state.attendance_marked = True
                                    st.session_state.marked_user = registered_name
                                    st.session_state.marked_time = time_str
                                    
                                    # Show BIG success message
                                    st.success("# ✅ ATTENDANCE MARKED SUCCESSFULLY!")
                                    st.balloons()
                                    
                                    # Show details in a highlighted box
                                    st.markdown(f"""
                                    <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #28a745;">
                                        <h2 style="color: #155724;">👤 {registered_name}</h2>
                                        <h3 style="color: #155724;">✅ Status: PRESENT</h3>
                                        <h3 style="color: #155724;">🕐 Time: {time_str}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show recent attendance immediately
                                    st.markdown("---")
                                    st.subheader("📋 Recent Attendance Log")
                                    recent_df = pd.read_csv(ATTENDANCE_FILE)
                                    recent_entries = recent_df.tail(10).sort_values(by=["Date", "Time"], ascending=False)
                                    st.dataframe(recent_entries, use_container_width=True, hide_index=True)
                                    
                                    # Force rerun to update status at top
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ **{registered_name}**, you've already marked attendance today!")
                                    
                                    # Show today's attendance
                                    st.markdown("---")
                                    st.subheader("📋 Today's Attendance")
                                    today_df = pd.read_csv(ATTENDANCE_FILE)
                                    today_entries = today_df[today_df["Date"] == date_str]
                                    if len(today_entries) > 0:
                                        st.dataframe(today_entries, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No attendance records for today yet.")
                                break
                                
                        except Exception as e:
                            continue
                    
                    if not found:
                        st.error("# ❌ FACE NOT RECOGNIZED!")
                        st.warning("Please try again or register first.")
                
                # Clean up temp file
                if os.path.exists(img_path):
                    os.remove(img_path)

# --------------------- Page 3: Biometric Log History ---------------------
elif page == "📊 Biometric Log History":
    st.title("📊 Biometric Log History")
    st.write("View complete attendance records and statistics.")
    
    df = pd.read_csv(ATTENDANCE_FILE)
    
    if not df.empty:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_entries = len(df)
            st.metric("Total Records", total_entries)
        
        with col2:
            unique_users = df["Name"].nunique()
            st.metric("Unique Users", unique_users)
        
        with col3:
            present_count = len(df[df["Status"] == "Present"])
            st.metric("Present", present_count)
        
        with col4:
            absent_count = len(df[df["Status"] == "Absent"])
            st.metric("Absent", absent_count)
        
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            name_filter = st.multiselect(
                "Filter by Name",
                options=sorted(df["Name"].unique()),
                default=None
            )
        
        with col2:
            date_filter = st.date_input("Filter by Date", value=None)
        
        with col3:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "Present", "Absent"]
            )
        
        # Apply filters
        filtered_df = df.copy()
        if name_filter:
            filtered_df = filtered_df[filtered_df["Name"].isin(name_filter)]
        if date_filter:
            filtered_df = filtered_df[filtered_df["Date"] == date_filter.strftime("%Y-%m-%d")]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]
        
        # Display table
        st.markdown("---")
        st.subheader(f"📋 Attendance Records ({len(filtered_df)} entries)")
        
        if len(filtered_df) > 0:
            st.dataframe(
                filtered_df.sort_values(by=["Date", "Time"], ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"attendance_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("📭 No records match the selected filters.")
        
    else:
        st.info("📭 No attendance records found. Start marking attendance!")
        st.write("Go to **Mark Attendance** page to record your first entry.")

# --------------------- Page 4: Manage Users ---------------------
elif page == "🗑️ Manage Users":
    st.title("🗑️ Manage Registered Users")
    st.write("View and delete registered users from the system.")
    st.warning("⚠️ **Warning:** Deleting a user will remove their photo AND all attendance records permanently!")
    
    registered_files = os.listdir(TRAIN_DIR)
    
    if registered_files:
        st.write(f"**Total Registered Users:** {len(registered_files)}")
        st.markdown("---")
        
        # Display users with delete buttons
        for file in sorted(registered_files):
            user_name = os.path.splitext(file)[0]
            user_path = os.path.join(TRAIN_DIR, file)
            
            # Check attendance count for this user
            df = pd.read_csv(ATTENDANCE_FILE)
            if not df.empty:
                user_attendance_count = len(df[df["Name"].apply(lambda x: normalize_name(x) == normalize_name(user_name))])
            else:
                user_attendance_count = 0
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"### 👤 {user_name}")
                st.write(f"📊 Attendance Records: {user_attendance_count}")
            
            with col2:
                # Show user image
                if st.button("👁️ View", key=f"view_{user_name}"):
                    st.session_state[f"show_{user_name}"] = not st.session_state.get(f"show_{user_name}", False)
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{user_name}", type="secondary"):
                    st.session_state[f"confirm_delete_{user_name}"] = True
            
            # Show image if view is clicked
            if st.session_state.get(f"show_{user_name}", False):
                try:
                    img = Image.open(user_path)
                    st.image(img, width=200, caption=user_name)
                except:
                    st.error("Could not load image")
            
            # Confirmation dialog for deletion
            if st.session_state.get(f"confirm_delete_{user_name}", False):
                st.error(f"⚠️ **CONFIRM DELETION**")
                st.write(f"**User:** {user_name}")
                st.write(f"**Attendance Records to be deleted:** {user_attendance_count}")
                st.write("This action cannot be undone!")
                
                col_confirm1, col_confirm2 = st.columns(2)
                
                with col_confirm1:
                    if st.button("✅ Yes, Delete Everything", key=f"confirm_yes_{user_name}", type="primary"):
                        success, message = delete_user_completely(user_name, user_path)
                        if success:
                            st.success(f"✅ {user_name} and {user_attendance_count} attendance records deleted successfully!")
                            st.session_state[f"confirm_delete_{user_name}"] = False
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                
                with col_confirm2:
                    if st.button("❌ Cancel", key=f"confirm_no_{user_name}"):
                        st.session_state[f"confirm_delete_{user_name}"] = False
                        st.rerun()
            
            st.markdown("---")
    else:
        st.info("📭 No registered users found.")
        st.write("Go to **Registration** page to register users.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Face Recognition Attendance System v1.0")
