"""
Export ChromaDB Contract Data for Power BI Integration
======================================================
Exports contract metadata to Parquet format (Power BI native) with automatic refresh capability.
Optimized for Power BI's Python connector with column types properly set.

Usage:
    python export_for_powerbi.py                    # Export to Parquet
    python export_for_powerbi.py --format csv        # Export to CSV
    python export_for_powerbi.py --refresh           # Scheduled refresh mode
"""

import chromadb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Power BI optimized column schema
POWERBI_SCHEMA = {
    # Identifiers
    'contract_id': 'string',
    'document_name': 'string',
    'file_path': 'string',
    
    # Parties & Entities
    'company_folder': 'string',
    'party_spectralink': 'string',
    'party_counterparty': 'string',
    'all_parties': 'string',
    
    # Contract Classification
    'contract_type': 'string',
    'document_role': 'string',
    'status': 'string',
    'terminated': 'bool',
    
    # Dates (crucial for time intelligence)
    'effective_date': 'datetime64[ns]',
    'expiration_date': 'datetime64[ns]',
    'termination_date': 'datetime64[ns]',
    'ingestion_date': 'datetime64[ns]',
    'days_until_expiration': 'int32',
    'is_expired': 'bool',
    'is_expiring_soon': 'bool',  # <90 days
    
    # Terms & Clauses
    'termination_for_convenience': 'string',
    'auto_renewal': 'string',
    'governing_law': 'string',
    'payment_terms': 'string',
    
    # Geography & Compliance
    'region': 'string',
    'gdpr_applicable': 'bool',
    'gdpr_reason': 'string',
    
    # Hierarchy & Relationships
    'parent_folder': 'string',
    'grandparent_folder': 'string',
    'sequence_label': 'string',
    'relationship_hints': 'string',
    
    # SharePoint Integration
    'sharepoint_owner': 'string',
    'sharepoint_department': 'string',
    
    # Metrics (for KPIs)
    'contract_value_usd': 'float64',
    'risk_score': 'int32',
    'text_length': 'int32',
    'page_count': 'int32',
}


class PowerBIExporter:
    """Export ChromaDB data in Power BI optimized format"""
    
    def __init__(self, chroma_path: str = "./chroma_db", output_dir: str = "./powerbi_exports"):
        self.chroma_path = Path(chroma_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"📊 Initializing Power BI Exporter")
        logger.info(f"   ChromaDB: {self.chroma_path}")
        logger.info(f"   Output: {self.output_dir}")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            self.collection = self.client.get_collection("contracts")
            logger.info(f"✅ Connected to ChromaDB collection: {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise
    
    def _parse_date(self, date_str: Any) -> pd.Timestamp:
        """Parse various date formats to pandas Timestamp"""
        if pd.isna(date_str) or date_str in ['', 'Unknown', 'Not specified', None]:
            return pd.NaT
        
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return pd.Timestamp(date_str)
        
        # Try parsing various formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%Y']:
            try:
                return pd.to_datetime(str(date_str), format=fmt)
            except:
                continue
        
        # Fallback: pandas smart parsing
        try:
            return pd.to_datetime(str(date_str))
        except:
            return pd.NaT
    
    def _extract_metadata_field(self, metadata: Dict, field: str, default: Any = None) -> Any:
        """Extract field from metadata with fallback"""
        value = metadata.get(field, default)
        if value == '' or value == 'Unknown':
            return default
        return value
    
    def _calculate_risk_score(self, row: Dict) -> int:
        """Calculate contract risk score (0-100) based on multiple factors"""
        score = 0
        
        # Expiration risk (40 points max)
        if pd.notna(row.get('expiration_date')):
            days_until = (pd.Timestamp(row['expiration_date']) - pd.Timestamp.now()).days
            if days_until < 0:
                score += 40  # Expired
            elif days_until < 30:
                score += 30  # < 30 days
            elif days_until < 90:
                score += 20  # < 90 days
            elif days_until < 180:
                score += 10  # < 180 days
        
        # Missing critical metadata (30 points max)
        if pd.isna(row.get('effective_date')):
            score += 10
        if row.get('party_counterparty') in [None, '', 'Unknown']:
            score += 10
        if row.get('governing_law') in [None, '', 'Unknown']:
            score += 10
        
        # GDPR risk (15 points)
        if row.get('gdpr_applicable') == True:
            score += 15
        
        # Termination risk (15 points)
        if row.get('termination_for_convenience') == 'Yes':
            score += 0  # Low risk - we can terminate
        elif row.get('termination_for_convenience') == 'No':
            score += 15  # High risk - locked in
        
        return min(score, 100)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export ChromaDB collection to Power BI optimized DataFrame"""
        logger.info("🔄 Fetching all contracts from ChromaDB...")
        
        # Get all documents with metadata
        results = self.collection.get(include=['metadatas', 'documents'])
        
        if not results['ids']:
            logger.warning("⚠️  No contracts found in ChromaDB")
            return pd.DataFrame()
        
        logger.info(f"✅ Retrieved {len(results['ids'])} contracts")
        
        # Build records list
        records = []
        for idx, contract_id in enumerate(results['ids']):
            metadata = results['metadatas'][idx]
            document = results['documents'][idx] if results['documents'] else ''
            
            # Parse dates
            effective_date = self._parse_date(metadata.get('effective_date'))
            expiration_date = self._parse_date(metadata.get('expiration_date'))
            termination_date = self._parse_date(metadata.get('termination_date'))
            ingestion_date = self._parse_date(metadata.get('ingestion_timestamp'))
            
            # Calculate derived fields
            days_until_expiration = None
            is_expired = False
            is_expiring_soon = False
            
            if pd.notna(expiration_date):
                days_until_expiration = (expiration_date - pd.Timestamp.now()).days
                is_expired = days_until_expiration < 0
                is_expiring_soon = 0 <= days_until_expiration <= 90
            
            # Build record - map ChromaDB field names to Power BI schema
            record = {
                'contract_id': contract_id,
                'document_name': self._extract_metadata_field(metadata, 'filename', 
                                self._extract_metadata_field(metadata, 'document_name', 'Unknown')),
                'file_path': self._extract_metadata_field(metadata, 'file_path', ''),
                
                # Company/Party fields - map from folder structure
                'company_folder': self._extract_metadata_field(metadata, 'parent_folder_name', 
                                 self._extract_metadata_field(metadata, 'company_folder', 'Unknown')),
                'party_spectralink': self._extract_metadata_field(metadata, 'party_spectralink', 'Spectralink'),
                'party_counterparty': self._extract_metadata_field(metadata, 'party_counterparty',
                                     self._extract_metadata_field(metadata, 'parent_folder_name', 'Unknown')),
                'all_parties': self._extract_metadata_field(metadata, 'parties', 
                              self._extract_metadata_field(metadata, 'all_parties', '')),
                
                # Contract identification
                'contract_type': self._extract_metadata_field(metadata, 'contract_type', 'Unknown'),
                'document_role': self._extract_metadata_field(metadata, 'document_role', 'Unknown'),
                'status': self._extract_metadata_field(metadata, 'contract_status', 
                         self._extract_metadata_field(metadata, 'status', 'Active')),
                'terminated': metadata.get('is_terminated', False) or metadata.get('terminated', 'No') == 'Yes',
                
                # Dates
                'effective_date': effective_date,
                'expiration_date': expiration_date,
                'termination_date': termination_date,
                'ingestion_date': ingestion_date or pd.Timestamp.now(),
                'days_until_expiration': days_until_expiration,
                'is_expired': is_expired,
                'is_expiring_soon': is_expiring_soon,
                
                # Terms & Clauses
                'termination_for_convenience': self._extract_metadata_field(metadata, 'termination_for_convenience', 'Unknown'),
                'auto_renewal': self._extract_metadata_field(metadata, 'auto_renewal', 'Unknown'),
                'governing_law': self._extract_metadata_field(metadata, 'governing_law', 'Unknown'),
                'payment_terms': self._extract_metadata_field(metadata, 'payment_terms', ''),
                
                # Geography & Compliance - map gdpr_applicable (string) to boolean
                'region': self._extract_metadata_field(metadata, 'region', 'Unknown'),
                'gdpr_applicable': metadata.get('gdpr_applicable', 'No') in ['Yes', 'True', True, '1', 1],
                'gdpr_reason': self._extract_metadata_field(metadata, 'gdpr_reason', ''),
                
                # Hierarchy - map field names
                'parent_folder': self._extract_metadata_field(metadata, 'parent_folder_name',
                                self._extract_metadata_field(metadata, 'parent_folder', '')),
                'grandparent_folder': self._extract_metadata_field(metadata, 'grandparent_folder_name',
                                     self._extract_metadata_field(metadata, 'grandparent_folder', '')),
                'sequence_label': self._extract_metadata_field(metadata, 'sequence_label', ''),
                'relationship_hints': self._extract_metadata_field(metadata, 'related_contract_hint',
                                     self._extract_metadata_field(metadata, 'relationship_hints', '')),
                
                # SharePoint Integration
                'sharepoint_owner': self._extract_metadata_field(metadata, 'sharepoint_owner_primary',
                                   self._extract_metadata_field(metadata, 'sharepoint_owner', '')),
                'sharepoint_department': self._extract_metadata_field(metadata, 'sharepoint_department', 'Legal'),
                
                # Metrics
                'contract_value_usd': 0.0,  # Placeholder for future enhancement
                'risk_score': 0,  # Will calculate below
                'text_length': int(metadata.get('char_count', len(document))) if document else 0,
                'page_count': int(metadata.get('page_count', 0)) if metadata.get('page_count') else 0,
            }
            
            # Calculate risk score
            record['risk_score'] = self._calculate_risk_score(record)
            
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Apply schema types
        logger.info("🔧 Applying Power BI optimized data types...")
        for col, dtype in POWERBI_SCHEMA.items():
            if col in df.columns:
                if 'datetime' in dtype:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'bool':
                    df[col] = df[col].astype(bool)
                elif 'int' in dtype:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                elif 'float' in dtype:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = df[col].astype(str).replace('nan', '')
        
        logger.info(f"✅ DataFrame created: {len(df)} rows × {len(df.columns)} columns")
        return df
    
    def export_to_parquet(self, filename: str = None) -> Path:
        """Export to Parquet format (Power BI native, best performance)"""
        df = self.export_to_dataframe()
        
        if df.empty:
            logger.warning("⚠️  No data to export")
            return None
        
        if filename is None:
            filename = f"contracts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        output_path = self.output_dir / filename
        
        logger.info(f"💾 Exporting to Parquet: {output_path}")
        df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Export complete: {file_size_mb:.2f} MB")
        
        return output_path
    
    def export_to_csv(self, filename: str = None) -> Path:
        """Export to CSV format (fallback for Power BI)"""
        df = self.export_to_dataframe()
        
        if df.empty:
            logger.warning("⚠️  No data to export")
            return None
        
        if filename is None:
            filename = f"contracts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = self.output_dir / filename
        
        logger.info(f"💾 Exporting to CSV: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')  # BOM for Excel
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Export complete: {file_size_mb:.2f} MB")
        
        return output_path
    
    def create_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for Power BI KPIs"""
        df = self.export_to_dataframe()
        
        if df.empty:
            return {}
        
        stats = {
            'total_contracts': len(df),
            'active_contracts': len(df[df['status'] == 'Active']),
            'terminated_contracts': len(df[df['terminated'] == True]),
            'expiring_soon': len(df[df['is_expiring_soon'] == True]),
            'expired_contracts': len(df[df['is_expired'] == True]),
            'avg_risk_score': df['risk_score'].mean(),
            'high_risk_contracts': len(df[df['risk_score'] >= 70]),
            'gdpr_contracts': len(df[df['gdpr_applicable'] == True]),
            'unique_companies': df['company_folder'].nunique(),
            'contracts_by_type': df['contract_type'].value_counts().to_dict(),
            'contracts_by_region': df['region'].value_counts().to_dict(),
            'last_update': datetime.now().isoformat(),
        }
        
        # Save summary
        summary_path = self.output_dir / 'summary_stats.json'
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Summary statistics saved: {summary_path}")
        return stats


def main():
    parser = argparse.ArgumentParser(description='Export ChromaDB contracts for Power BI')
    parser.add_argument('--format', choices=['parquet', 'csv', 'both'], default='parquet',
                       help='Export format (default: parquet)')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--refresh', action='store_true',
                       help='Scheduled refresh mode (overwrites existing file)')
    parser.add_argument('--chroma-path', type=str, default='./chroma_db',
                       help='Path to ChromaDB directory')
    parser.add_argument('--output-dir', type=str, default='./powerbi_exports',
                       help='Output directory for exports')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = PowerBIExporter(chroma_path=args.chroma_path, output_dir=args.output_dir)
    
    # Export in specified format
    if args.format in ['parquet', 'both']:
        filename = args.output if args.output else ('contracts_latest.parquet' if args.refresh else None)
        exporter.export_to_parquet(filename)
    
    if args.format in ['csv', 'both']:
        filename = args.output if args.output else ('contracts_latest.csv' if args.refresh else None)
        exporter.export_to_csv(filename)
    
    # Generate summary statistics
    stats = exporter.create_summary_stats()
    
    logger.info("=" * 70)
    logger.info("📈 EXPORT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total Contracts: {stats.get('total_contracts', 0):,}")
    logger.info(f"Active Contracts: {stats.get('active_contracts', 0):,}")
    logger.info(f"Expiring Soon (<90 days): {stats.get('expiring_soon', 0):,}")
    logger.info(f"Expired: {stats.get('expired_contracts', 0):,}")
    logger.info(f"High Risk (≥70): {stats.get('high_risk_contracts', 0):,}")
    logger.info(f"GDPR Applicable: {stats.get('gdpr_contracts', 0):,}")
    logger.info(f"Unique Companies: {stats.get('unique_companies', 0):,}")
    logger.info("=" * 70)
    logger.info("✅ Power BI export complete!")


if __name__ == '__main__':
    main()
