# Power Automate Flow Templates
## Contract Management Automation Workflows

These flows connect your contract data to Microsoft 365 services for automated notifications, approvals, and data entry.

---

## 🎯 **Overview of Flows**

1. **Expiration Alert Flow** - Send Outlook emails for contracts expiring soon
2. **Contract Intake Flow** - Microsoft Forms → SharePoint → Python ingestion
3. **Risk Score Alert Flow** - Notify when contracts exceed risk threshold
4. **Weekly Executive Summary** - Automated email digest
5. **SharePoint Upload Trigger** - Auto-ingest new contracts

---

## 📧 **Flow 1: Contract Expiration Alerts**

### **Trigger:** Recurrence (Daily at 8 AM)

### **Steps:**

1. **Initialize Variables**
   - `varToday` (Date) = `utcNow()`
   - `varThreshold` (Integer) = `90` (days)
   - `varEmailBody` (String) = `''`

2. **Run Python Script** (via HTTP or Azure Function)
   ```python
   # Call contract_expiration_checker.py
   GET https://your-azure-function-url/api/check-expirations?days=90
   ```

3. **Parse JSON Response**
   ```json
   {
     "expiring_contracts": [
       {
         "contract_id": "contract_xxx",
         "document_name": "MSA with Company X",
         "company": "Company X",
         "expiration_date": "2026-02-15",
         "days_until_expiration": 82,
         "risk_score": 65,
         "sharepoint_owner": "wayne.trout@spectralink.com"
       }
     ]
   }
   ```

4. **Apply to Each** Contract
   
   a. **Append to String Variable** (`varEmailBody`)
   ```html
   <tr>
     <td>@{items('Apply_to_each')?['document_name']}</td>
     <td>@{items('Apply_to_each')?['company']}</td>
     <td>@{items('Apply_to_each')?['expiration_date']}</td>
     <td>@{items('Apply_to_each')?['days_until_expiration']} days</td>
     <td style="background-color: @{if(greater(items('Apply_to_each')?['risk_score'], 70), '#F44336', '#4CAF50')}">
       @{items('Apply_to_each')?['risk_score']}
     </td>
   </tr>
   ```

5. **Condition:** If expiring contracts > 0

6. **Send Email (Outlook)**
   - **To:** `@{items('Apply_to_each')?['sharepoint_owner']}`
   - **CC:** `legal@spectralink.com`
   - **Subject:** `⚠️ Contract Expiration Alert - @{items('Apply_to_each')?['company']}`
   - **Body (HTML):**
   ```html
   <html>
   <head>
     <style>
       body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f4; }
       .container { max-width: 800px; margin: 20px auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
       .header { background: #00BCF2; color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }
       .alert-high { background-color: #F44336; color: white; padding: 10px; border-radius: 4px; margin: 15px 0; }
       .alert-medium { background-color: #FF9800; color: white; padding: 10px; border-radius: 4px; margin: 15px 0; }
       table { width: 100%; border-collapse: collapse; margin: 20px 0; }
       th { background-color: #333; color: white; padding: 12px; text-align: left; }
       td { padding: 12px; border-bottom: 1px solid #ddd; }
       .footer { text-align: center; color: #666; margin-top: 20px; font-size: 12px; }
     </style>
   </head>
   <body>
     <div class="container">
       <div class="header">
         <h1>🚨 Contract Expiration Alert</h1>
         <p>The following contracts are expiring within @{variables('varThreshold')} days</p>
       </div>
       
       <div class="alert-high">
         <strong>Action Required:</strong> Please review these contracts and initiate renewal or termination procedures.
       </div>
       
       <table>
         <thead>
           <tr>
             <th>Contract Name</th>
             <th>Company</th>
             <th>Expiration Date</th>
             <th>Days Remaining</th>
             <th>Risk Score</th>
           </tr>
         </thead>
         <tbody>
           @{variables('varEmailBody')}
         </tbody>
       </table>
       
       <div class="footer">
         <p>This is an automated message from the Contract Management System.</p>
         <p>For questions, contact <a href="mailto:legal@spectralink.com">legal@spectralink.com</a></p>
       </div>
     </div>
   </body>
   </html>
   ```

---

## 📝 **Flow 2: Contract Intake (Microsoft Forms)**

### **Trigger:** When a new response is submitted (Microsoft Forms)

### **Form Questions:**
1. Company Name (Text)
2. Contract Type (Choice: NDA, MSA, SOW, Amendment, Other)
3. Upload Contract File (File Upload)
4. Effective Date (Date)
5. Expiration Date (Date)
6. Region (Choice: AMER, EMEA, APAC)
7. Notes (Long Text)

### **Steps:**

1. **Get Response Details** (Forms)
   - Form ID: `{your-form-id}`
   - Response ID: `{triggerOutputs()?['body/resourceData/responseId']}`

2. **Get File Content** (Forms)
   - File ID from response

3. **Create File** (SharePoint)
   - Site: `https://spectralink.sharepoint.com/sites/Contracts`
   - Folder: `/Contracts/Intake/@{outputs('Get_response_details')?['body/r_company_name']}`
   - File Name: `@{outputs('Get_response_details')?['body/r_contract_file']['name']}`
   - File Content: `@{outputs('Get_file_content')?['body']}`

4. **Create Item** (SharePoint List - "Contract Intake Queue")
   - Company: `@{outputs('Get_response_details')?['body/r_company_name']}`
   - Contract Type: `@{outputs('Get_response_details')?['body/r_contract_type']}`
   - Effective Date: `@{outputs('Get_response_details')?['body/r_effective_date']}`
   - Expiration Date: `@{outputs('Get_response_details')?['body/r_expiration_date']}`
   - Region: `@{outputs('Get_response_details')?['body/r_region']}`
   - File URL: `@{outputs('Create_file')?['ItemId']}`
   - Status: `Pending Processing`
   - Submitted By: `@{outputs('Get_response_details')?['body/responder']}`

5. **HTTP Request** (Trigger Python Ingestion)
   ```
   POST https://your-azure-function-url/api/ingest-contract
   Headers:
     Content-Type: application/json
   Body:
   {
     "file_path": "@{outputs('Create_file')?['Path']}",
     "company_name": "@{outputs('Get_response_details')?['body/r_company_name']}",
     "contract_type": "@{outputs('Get_response_details')?['body/r_contract_type']}",
     "effective_date": "@{outputs('Get_response_details')?['body/r_effective_date']}",
     "expiration_date": "@{outputs('Get_response_details')?['body/r_expiration_date']}",
     "region": "@{outputs('Get_response_details')?['body/r_region']}"
   }
   ```

6. **Update Item** (SharePoint)
   - Status: `Processed`
   - Ingestion Date: `@{utcNow()}`
   - ChromaDB ID: `@{outputs('HTTP')?['body/contract_id']}`

7. **Send Email** (Outlook - Confirmation to Submitter)
   - To: `@{outputs('Get_response_details')?['body/responder']}`
   - Subject: `✅ Contract Submitted Successfully - @{outputs('Get_response_details')?['body/r_company_name']}`
   - Body:
   ```html
   <p>Your contract has been successfully submitted and processed.</p>
   <ul>
     <li><strong>Company:</strong> @{outputs('Get_response_details')?['body/r_company_name']}</li>
     <li><strong>Type:</strong> @{outputs('Get_response_details')?['body/r_contract_type']}</li>
     <li><strong>File:</strong> @{outputs('Get_response_details')?['body/r_contract_file']['name']}</li>
     <li><strong>Contract ID:</strong> @{outputs('HTTP')?['body/contract_id']}</li>
   </ul>
   <p>You can view the contract in <a href="@{outputs('Create_file')?['LinkingUri']}">SharePoint</a>.</p>
   ```

---

## 📊 **Flow 3: Weekly Executive Summary**

### **Trigger:** Recurrence (Weekly on Monday at 7 AM)

### **Steps:**

1. **HTTP Request** (Get Summary Stats)
   ```
   GET https://your-azure-function-url/api/get-summary-stats
   ```

2. **Parse JSON** (Summary Response)
   ```json
   {
     "total_contracts": 1241,
     "active_contracts": 987,
     "expiring_soon": 45,
     "expired_contracts": 23,
     "high_risk_contracts": 12,
     "avg_risk_score": 42.5,
     "gdpr_contracts": 234,
     "unique_companies": 156,
     "contracts_by_type": {"NDA": 456, "MSA": 234, "SOW": 321, "Other": 230},
     "top_expiring": [
       {"company": "Company A", "days": 15},
       {"company": "Company B", "days": 23}
     ]
   }
   ```

3. **Send Email** (Outlook - Executive Team)
   - To: `executives@spectralink.com`
   - CC: `legal@spectralink.com; wayne.trout@spectralink.com`
   - Subject: `📊 Weekly Contract Management Summary - Week of @{formatDateTime(utcNow(), 'MM/dd/yyyy')}`
   - Importance: High
   - Body (HTML): See full template in file

---

## 🔔 **Flow 4: High Risk Alert**

### **Trigger:** When item is created or modified (SharePoint List - from Python script updates)

### **Condition:** `risk_score >= 70`

### **Steps:**

1. **Get File Properties** (SharePoint)
   - Site: Contract library
   - ID: `@{triggerOutputs()?['body/ID']}`

2. **Send Email** (Outlook - Risk Alert)
   - To: `legal@spectralink.com; risk-management@spectralink.com`
   - Subject: `🔴 HIGH RISK CONTRACT ALERT - @{outputs('Get_file_properties')?['body/{FilenameWithExtension}']}`
   - Priority: High
   - Body:
   ```html
   <h2 style="color: #F44336;">⚠️ High Risk Contract Detected</h2>
   <p>A contract has been flagged as high risk (score: @{outputs('Get_file_properties')?['body/risk_score']}).</p>
   
   <h3>Contract Details:</h3>
   <ul>
     <li><strong>Company:</strong> @{outputs('Get_file_properties')?['body/company_folder']}</li>
     <li><strong>Type:</strong> @{outputs('Get_file_properties')?['body/contract_type']}</li>
     <li><strong>Expiration:</strong> @{outputs('Get_file_properties')?['body/expiration_date']}</li>
     <li><strong>Days Until Expiration:</strong> @{outputs('Get_file_properties')?['body/days_until_expiration']}</li>
     <li><strong>Risk Score:</strong> @{outputs('Get_file_properties')?['body/risk_score']}/100</li>
   </ul>
   
   <p><a href="@{outputs('Get_file_properties')?['body/{Link}']}" style="background-color: #00BCF2; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Contract in SharePoint</a></p>
   ```

---

## 🔄 **Flow 5: Auto-Ingest on SharePoint Upload**

### **Trigger:** When a file is created (SharePoint - Contracts Library)

### **Condition:** File extension is `.pdf` or `.docx`

### **Steps:**

1. **Get File Content** (SharePoint)

2. **HTTP Request** (Trigger Python Ingestion)
   ```
   POST https://your-azure-function-url/api/ingest-contract
   Body:
   {
     "file_path": "@{triggerOutputs()?['body/{FullPath}']}",
     "file_content_base64": "@{base64(outputs('Get_file_content')?['body'])}",
     "uploaded_by": "@{triggerOutputs()?['body/Author/Email']}"
   }
   ```

3. **Update File Properties** (SharePoint)
   - Ingestion Status: `Processed`
   - ChromaDB ID: `@{outputs('HTTP')?['body/contract_id']}`
   - Processed Date: `@{utcNow()}`

---

## 🚀 **Deployment Instructions**

### **Option A: Manual Creation in Power Automate**
1. Go to [make.powerautomate.com](https://make.powerautomate.com)
2. Click **+ Create** → **Automated cloud flow**
3. Follow step-by-step instructions for each flow above
4. Test each flow before enabling

### **Option B: Import Flow Package** (if you have .zip exports)
1. Export each flow as package
2. Share package files with team
3. Import via **My flows** → **Import** → **Upload**

### **Azure Function Setup** (for HTTP triggers):
See `AZURE_FUNCTIONS_SETUP.md` for deploying Python ingestion endpoints

---

## 📋 **Flow Summary Table**

| Flow Name | Trigger | Frequency | Key Actions | Recipients |
|-----------|---------|-----------|-------------|------------|
| Expiration Alerts | Recurrence | Daily 8 AM | Check expirations, send emails | SharePoint owners |
| Contract Intake | Forms submission | On-demand | Upload to SharePoint, ingest to ChromaDB | Submitter |
| Risk Alerts | SharePoint update | On-demand | Detect high risk, send alert | Legal, Risk Mgmt |
| Executive Summary | Recurrence | Weekly Mon 7 AM | Aggregate stats, send report | Executives, Legal |
| Auto-Ingest | SharePoint upload | On-demand | Process new files | N/A (silent) |

---

## 💡 **Customization Tips**

1. **Change Email Templates:** Edit HTML in Send Email actions
2. **Adjust Thresholds:** Modify `varThreshold` in expiration flow
3. **Add Approvals:** Insert "Start and wait for approval" action
4. **Teams Notifications:** Replace Outlook with "Post message in Teams"
5. **Conditional Routing:** Add more conditions for different stakeholders

---

## 🐛 **Troubleshooting**

**Flow fails on HTTP request:**
- Check Azure Function URL is correct
- Verify authentication headers
- Test endpoint manually with Postman

**Email not sending:**
- Verify Outlook connector permissions
- Check recipient email addresses
- Review email size limits (<25 MB)

**SharePoint upload fails:**
- Check folder permissions
- Verify site URL is correct
- Ensure file size < 250 MB

---

**Next:** See `GOOGLE_ADK_INTEGRATION.md` for AI agent setup
