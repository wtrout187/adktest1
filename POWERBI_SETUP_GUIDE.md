# Power BI Dashboard Setup Guide
## Contract Management Analytics with Custom Visuals

This guide walks you through creating a professional dark-themed Power BI dashboard with custom Python visuals, DAX measures, and Power Automate integration.

---

## 📁 **Prerequisites**

1. **Power BI Desktop** (latest version) - Download from [powerbi.microsoft.com](https://powerbi.microsoft.com)
2. **Python 3.8+** installed and configured in Power BI
3. **Required Python packages**:
   ```powershell
   pip install pandas numpy matplotlib seaborn networkx pyarrow
   ```
4. **Data exported from ChromaDB**: Run `python export_for_powerbi.py`

---

## 🎨 **Step 1: Apply Dark Theme**

1. Open Power BI Desktop
2. Go to **View** tab → **Themes** → **Browse for themes**
3. Select `powerbi_theme.json` from your project folder
4. Click **Apply** → Your entire report will adopt the dark theme

---

## 📊 **Step 2: Import Contract Data**

### Option A: Parquet (Recommended - Fastest)
1. **Home** tab → **Get Data** → **More...**
2. Search for **Parquet**
3. Navigate to `powerbi_exports\contracts_latest.parquet`
4. Click **Load**

### Option B: CSV (Fallback)
1. **Home** tab → **Get Data** → **Text/CSV**
2. Select `powerbi_exports\contracts_latest.csv`
3. Power BI will auto-detect types → Click **Load**

### Option C: Python Script (Dynamic Refresh)
1. **Home** tab → **Get Data** → **More...** → **Python script**
2. Paste this code:
```python
import sys
sys.path.append(r'C:\Users\wtrout\Python Projects\ADKTest')
from export_for_powerbi import PowerBIExporter

exporter = PowerBIExporter(chroma_path="./chroma_db")
dataset = exporter.export_to_dataframe()
```
3. Click **OK** → Select `dataset` table → **Load**

---

## 🧮 **Step 3: Create DAX Measures**

1. In **Data** view, click your `Contracts` table
2. **Modeling** tab → **New Measure**
3. Copy measures from `powerbi_dax_measures.txt` one at a time
4. Name each measure as shown in the file (e.g., `Total Contracts`, `Active Contracts`, etc.)

**Quick Start - Create these 5 measures first:**
```dax
Total Contracts = COUNTROWS(Contracts)

Active Contracts = 
CALCULATE(
    COUNTROWS(Contracts),
    Contracts[status] = "Active",
    Contracts[is_expired] = FALSE
)

Expiring Soon = 
CALCULATE(
    COUNTROWS(Contracts),
    Contracts[is_expiring_soon] = TRUE
)

Avg Risk Score = AVERAGE(Contracts[risk_score])

Critical Alerts = 
VAR ExpiredCritical = [Expired Contracts]
VAR ExpiringCritical = 
    CALCULATE(
        COUNTROWS(Contracts),
        Contracts[days_until_expiration] <= 30,
        Contracts[is_expired] = FALSE
    )
RETURN
ExpiredCritical + ExpiringCritical
```

---

## 📈 **Step 4: Build Dashboard Pages**

### **Page 1: Executive Overview** 📋

#### KPI Cards (Top Row):
1. Drag **Card** visual to canvas
2. Add `[Total Contracts]` to **Fields**
3. **Format** → Set font size to 32px, bold
4. Repeat for:
   - `[Active Contracts]` (green background)
   - `[Expiring Soon]` (yellow background)
   - `[Avg Risk Score]` (conditional: red if >70, green if <40)
   - `[Critical Alerts]` (red background)

#### Main Charts:
1. **Line Chart**: Contract Expiration Timeline
   - X-axis: `expiration_date` (Month hierarchy)
   - Y-axis: Count of `contract_id`
   - Legend: `status`

2. **Donut Chart**: Status Distribution
   - Legend: `status`
   - Values: Count of `contract_id`
   - Enable data labels

3. **Clustered Bar**: Top 10 Companies
   - Y-axis: `company_folder`
   - X-axis: Count of `contract_id`
   - Filter: `[Top 10 Companies] = 1` (DAX measure)

4. **Matrix**: Risk Heatmap
   - Rows: `company_folder`
   - Columns: `contract_type`
   - Values: `[Avg Risk Score]`
   - Format → Conditional formatting → Background color (gradient: green→yellow→red)

---

### **Page 2: Expiration Timeline** 📅

#### Python Visual: Gantt Timeline
1. Insert **Python visual**
2. Drag these fields to **Values**:
   - `document_name`
   - `effective_date`
   - `expiration_date`
   - `status`
   - `risk_score`
3. In Python script editor, paste from `powerbi_visual_generators.py`:
   ```python
   # Copy the powerbi_contract_timeline() function code
   ```
4. Click **Run** (play button)

#### Table: Contracts Expiring Soon
1. Insert **Table** visual
2. Add columns:
   - `document_name`
   - `company_folder`
   - `expiration_date`
   - `days_until_expiration`
   - `risk_score`
3. Filter: `is_expiring_soon = TRUE`
4. Format → Conditional formatting on `days_until_expiration` (red <30, yellow 30-60, green >60)

---

### **Page 3: Network Analysis** 🕸️

#### Python Visual: Party Network Graph
1. Insert **Python visual**
2. Drag fields:
   - `party_spectralink`
   - `party_counterparty`
   - `contract_type`
   - `company_folder`
3. Paste `powerbi_party_network()` function from `powerbi_visual_generators.py`
4. Run script

This creates an interactive network showing relationships between Spectralink entities and counterparties.

---

### **Page 4: Risk Analysis** ⚠️

#### Python Visual: Risk Heatmap
1. Insert **Python visual**
2. Drag fields:
   - `company_folder`
   - `contract_type`
   - `risk_score`
3. Paste `powerbi_risk_heatmap()` function
4. Run script

#### High Risk Table:
1. Insert **Table** visual
2. Add all relevant columns
3. Filter: `risk_score >= 70`
4. Sort by `risk_score` descending

---

### **Page 5: GDPR Compliance** 🔒

#### Python Visual: GDPR Treemap
1. Insert **Python visual**
2. Drag fields:
   - `region`
   - `gdpr_applicable`
   - `contract_id` (count)
3. Paste `powerbi_gdpr_treemap()` function
4. Run script

#### GDPR Table:
1. Filter entire page to `gdpr_applicable = TRUE`
2. Add table with GDPR contracts
3. Group by `region`

---

## 🎛️ **Step 5: Add Interactive Slicers**

Place these slicers on **every page** (sync them across pages):

1. **Region** (multi-select dropdown)
2. **Contract Type** (multi-select dropdown)
3. **Status** (multi-select, default: Active)
4. **Company Folder** (search-enabled dropdown)
5. **Date Range** (relative date slicer on `expiration_date`)

**To sync slicers across pages:**
1. Select a slicer → **View** tab → **Sync slicers**
2. Check all pages where it should appear
3. Check "Sync field" to link filtering

---

## 🔖 **Step 6: Create Bookmarks (Navigation)**

1. **View** tab → **Bookmarks** pane
2. Set up a page, then click **Add** bookmark
3. Create bookmarks for:
   - "Overview" (Page 1, no filters)
   - "Expiring Soon" (Page 2, filtered to `is_expiring_soon = TRUE`)
   - "High Risk" (Page 4, filtered to `risk_score >= 70`)
   - "GDPR" (Page 5, filtered to `gdpr_applicable = TRUE`)

4. Add **Buttons** for navigation:
   - Insert → Button → Blank
   - Set **Text** to bookmark name
   - **Action** → Type: Bookmark → Select bookmark
   - Style with dark theme colors

---

## 🔄 **Step 7: Configure Scheduled Refresh**

### For Power BI Service (Cloud):
1. Publish report to Power BI Service
2. In workspace, click **...** on dataset → **Settings**
3. **Scheduled refresh** → Enable
4. Set frequency (e.g., Daily at 6 AM)
5. Configure gateway if using local files

### For Python-based refresh:
1. Create a **Windows Task Scheduler** job:
   ```powershell
   python export_for_powerbi.py --refresh --format parquet
   ```
2. Schedule to run before Power BI refresh (e.g., 5:30 AM)
3. Power BI will automatically reload the updated Parquet file

---

## 📱 **Step 8: Optimize for Mobile**

1. **View** tab → **Mobile layout**
2. Drag visuals to mobile canvas
3. Priority order:
   - KPI cards at top
   - Critical alerts card
   - Expiring soon table
   - Main charts

---

## 🚀 **Step 9: Publish & Share**

1. **Home** tab → **Publish**
2. Select workspace (create "Contract Management" workspace if needed)
3. Once published, click **Open in Power BI**
4. Create **Dashboard** by pinning key visuals
5. Share with stakeholders:
   - Click **Share** → Enter email addresses
   - Set permissions (View only / Edit)
   - Enable "Allow recipients to share"

---

## 🔗 **Step 10: Power Automate Integration**

### See `power_automate_flows.md` for:
- Expiration email alerts (Outlook)
- Contract intake workflow (Forms → SharePoint → Python)
- Risk score change notifications
- Weekly executive summary emails

---

## 💡 **Pro Tips**

### Performance Optimization:
- Use **Parquet** format instead of CSV (10x faster loading)
- Create **aggregations** in Power BI for large datasets (>1M rows)
- Use **DirectQuery** instead of Import if data changes frequently

### Visual Best Practices:
- Limit to 5-7 visuals per page (avoid clutter)
- Use consistent color scheme (dark theme colors)
- Add **tooltips** with additional context
- Enable **cross-filtering** between visuals

### Python Visuals:
- Python visuals don't support interactivity (clicks won't filter)
- Pre-process data in DAX when possible (faster than Python)
- Use Python for complex visualizations not available in Power BI

### Mobile Optimization:
- Test mobile layout on phone before publishing
- Use **card visuals** for KPIs (better mobile rendering)
- Avoid complex Python visuals on mobile (slow rendering)

---

## 🐛 **Troubleshooting**

**Python visuals not showing:**
1. File → Options → Python scripting → Set Python home directory
2. Install required packages in that Python environment
3. Restart Power BI Desktop

**Data not refreshing:**
1. Check file paths are absolute, not relative
2. Verify ChromaDB path is correct in Python script
3. Test export script manually: `python export_for_powerbi.py`

**DAX errors:**
1. Ensure column names match exactly (case-sensitive)
2. Check for calculated columns vs measures
3. Use `CALCULATE` to change filter context

**Theme not applying:**
1. Check JSON syntax (use JSON validator)
2. Ensure no trailing commas in JSON
3. Re-import theme after fixing errors

---

## 📚 **Additional Resources**

- [Power BI Documentation](https://docs.microsoft.com/power-bi/)
- [DAX Guide](https://dax.guide/)
- [Python Visuals Documentation](https://docs.microsoft.com/power-bi/visuals/service-python-visuals-support)
- [Power BI Community](https://community.powerbi.com/)

---

## ✅ **Checklist**

- [ ] Exported data from ChromaDB
- [ ] Applied dark theme
- [ ] Imported contract data
- [ ] Created all DAX measures
- [ ] Built 5 dashboard pages
- [ ] Added synchronized slicers
- [ ] Created navigation bookmarks
- [ ] Configured scheduled refresh
- [ ] Optimized for mobile
- [ ] Published to Power BI Service
- [ ] Set up Power Automate flows
- [ ] Shared with stakeholders

---

**Next Steps:** See `power_automate_flows.md` for workflow automation and `google_adk_integration.md` for AI agent setup.
