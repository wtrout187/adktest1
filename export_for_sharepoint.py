#!/usr/bin/env python3
"""
Export ChromaDB Contract Data to SharePoint List Format
========================================================
Exports contract metadata to Excel format matching SharePoint list schema.
This aligns our extracted data with your existing SharePoint Contracts list.

Your SharePoint List Columns (from sharepoint list example.xlsx):
-----------------------------------------------------------------
Name, Content Type, ContractDuration, Party1 Name, Party1 Address, Party1 ReferenceName,
Party2 Name, Party2 Address, Party2 ReferenceName, Jurisdiction, ContractTitle,
ExecutionDate, EffectiveDate, RenewalDate, ContractId, ExpirationDate, Processed,
Processing status, Modified, Modified By, Autorenew, Notification Period in days,
Audit Flag, Lifecycle Stage, Risk Level, Archive Flag, GDPR_Applicable, Contract_Owner,
NotificationPeriodDays, Notification_Buffer_Days, Department, ContractValue, ContractSummary

Usage:
    python export_for_sharepoint.py                    # Export to Excel
    python export_for_sharepoint.py --preview 10       # Preview first 10 rows
    python export_for_sharepoint.py --validate         # Compare with SharePoint schema
"""

import chromadb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
import os
import re
from typing import Dict, Any, List, Optional, Tuple, Set, cast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SharePointExporter:
    """Export ChromaDB data in SharePoint list format"""
    EMEA_KEYWORDS = [
        'united kingdom', 'england', 'scotland', 'wales', 'ireland', 'denmark', 'germany',
        'france', 'spain', 'portugal', 'sweden', 'norway', 'finland', 'switzerland', 'austria',
        'italy', 'belgium', 'netherlands', 'luxembourg', 'poland', 'czech', 'slovak', 'hungary',
        'romania', 'bulgaria', 'slovenia', 'croatia', 'serbia', 'estonia', 'latvia', 'lithuania',
        'iceland', 'greece'
    ]
    NA_APAC_REGION_KEYS = {'north america', 'na', 'apac', 'asia pac', 'asia-pacific', 'asia pacific'}
    NOTICE_CANDIDATE_FIELDS = [
        'termination_notice',
        'termination_clause',
        'termination_rights',
        'renewal_notice',
        'notice_period',
        'notice_clause'
    ]
    INITIAL_TERM_FIELDS = ['initial_term', 'initial_term_text', 'term_initial', 'term_clause', 'contract_term']
    RENEWAL_TERM_FIELDS = ['renewal_term', 'renewal_clause', 'renewal_terms', 'renewal_provisions']
    TERMINATION_NOTICE_FIELDS = ['termination_notice_period', 'termination_notice', 'termination_clause', 'termination_rights', 'termination_provision']
    RENEWAL_NOTICE_FIELDS = ['renewal_notice_period', 'renewal_notice', 'notice_period', 'renewal_notification']
    PERPETUAL_NOTICE_FIELDS = ['perpetual_notice_period', 'perpetual_notice', 'perpetual_termination_notice', 'evergreen_notice']
    EXECUTION_DATE_FIELDS = ['execution_date', 'signature_date', 'sign_date', 'signed_date']
    EFFECTIVE_DATE_FIELDS = ['effective_date', 'effective_date_extracted', 'start_date', 'commencement_date', 'sow_start_date']
    EXPIRATION_DATE_FIELDS = ['expiration_date', 'expiry_date', 'end_date', 'termination_date', 'sow_end_date']
    RENEWAL_DATE_FIELDS = ['renewal_date', 'next_renewal_date']
    
    # SharePoint list schema (matching your example)
    SHAREPOINT_SCHEMA = {
        'Name': 'string',                          # Document filename
        'Content Type': 'string',                  # "WRT Global B2B Contract Processing"
        'ContractDuration': 'string',              # "01/19/2026 - 01/21/2026"
        'Party1 Name': 'string',                   # Counterparty company name
        'Party1 Address': 'string',                # Counterparty address
        'Party1 ReferenceName': 'string',          # "Hotel", "Client", "Vendor" etc
        'Party2 Name': 'string',                   # Spectralink entity
        'Party2 Address': 'string',                # Spectralink address
        'Party2 ReferenceName': 'string',          # "Client" (typically Spectralink's role)
        'Jurisdiction': 'string',                  # Governing law/country
        'ContractTitle': 'string',                 # Full contract title
        'ExecutionDate': 'datetime64[ns]',         # Date signed
        'EffectiveDate': 'datetime64[ns]',         # Start date
        'RenewalDate': 'datetime64[ns]',           # Next renewal
        'ContractId': 'string',                    # Unique ID
        'ExpirationDate': 'datetime64[ns]',        # End date
        'Processed': 'datetime64[ns]',             # Processing timestamp
        'Processing status': 'string',             # "Finished", "Pending", "Error"
        'Modified': 'datetime64[ns]',              # Last modified
        'Modified By': 'string',                   # Who modified
        'Autorenew': 'string',                     # "Yes"/"No"
        'Notification Period in days': 'int64',    # Notice period (30, 60, 90, etc)
        'Audit Flag': 'string',                    # "Yes"/"No"
        'Lifecycle Stage': 'string',               # "Active", "Review", "Expired", "Terminated"
        'Risk Level': 'string',                    # "Low", "Medium", "High"
        'Archive Flag': 'string',                  # "Yes"/"No"
        'GDPR_Applicable': 'string',               # "Yes"/"No"
        'Contract_Owner': 'string',                # SharePoint user reference
        'NotificationPeriodDays': 'int64',         # Notification period (duplicate?)
        'Notification_Buffer_Days': 'int64',       # Buffer before notification
        'Department': 'string',                    # Business department
        'ContractValue': 'string',                 # "$50,000" or "£27,335"
        'ContractSummary': 'string',               # AI-generated summary
        # Contract Family / Hierarchy Fields
        'ContractFamily': 'string',                # Family ID (e.g., "Matellio-2015")
        'ParentContractId': 'string',              # Links SOW/Amendment to governing MSA
        'ContractHierarchy': 'string',             # "MSA", "SOW", "Amendment", "NDA", "Standalone"
        'DaysToExpiry': 'int64',                   # Calculated days until expiration
        'TermsInherited': 'string',                # "Yes"/"No" - indicates terms from parent
        'SOWStartDate': 'datetime64[ns]',          # SOW-specific start date
        'SOWEndDate': 'datetime64[ns]',            # SOW-specific end date
        'ParentAgreementDate': 'string',           # Reference date for parent MSA
        'ContractIdNotes': 'string',               # Flags for ID generation issues
        'InitialTerm': 'string',                   # Parsed initial term text
        'RenewalTerm': 'string',                   # Parsed renewal term text
        'TerminationNotice': 'string',             # Raw termination notice clause
        'RenewalNotice': 'string',                 # Raw renewal notice clause
        'PerpetualAgreement': 'string',            # Yes/No for perpetual contracts
        'PerpetualNotice': 'string',               # Raw perpetual notice text
        'PerpetualNoticeDays': 'int64',            # Parsed perpetual notice in days
        'GDPRReason': 'string',                    # Why GDPR applies
        'AuditReason': 'string',                   # Why audit flag triggered
        'ArchiveReason': 'string',                 # Why archive flag triggered
        'Contract_CoOwner': 'string',              # Secondary owner / co-owner
        'FolderStatusNotes': 'string',             # Folder-based observations
        'ReviewFlag': 'string',                    # Draft/review status guidance
        'CounterpartyRegion': 'string',            # Detected counterparty region
    }
    
    def __init__(self, chroma_path: str = "./chroma_db", output_dir: str = "./sharepoint_exports", collection_name: str = "contracts"):
        self.chroma_path = Path(chroma_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collection_name = collection_name
        
        logger.info(f"📊 Initializing SharePoint Exporter")
        logger.info(f"   ChromaDB: {self.chroma_path}")
        logger.info(f"   Collection: {self.collection_name}")
        logger.info(f"   Output: {self.output_dir}")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"✅ Connected to ChromaDB collection: {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"❌ Failed to connect to ChromaDB: {e}")
            raise

        self.default_owner = os.getenv('CLM_DEFAULT_CONTRACT_OWNER', 'wayne.trout@spectralink.com')
        self.default_co_owner = os.getenv('CLM_DEFAULT_CONTRACT_CO_OWNER', self.default_owner)
        self.distribution_owner = os.getenv('CLM_OWNER_NA_APAC_DISTRIBUTION', 'tom.roberts@spectralink.com')
        
        # Baseline export schema (contracts_sharepoint_complete.xlsx) - order matters
        self.BASELINE_EXPORT_COLUMNS = [
            'Name', 'Content Type', 'ContractId', 'ContractTitle', 'ContractDuration',
            'Party1 Name', 'Party1 Address', 'Party2 Name', 'Party2 Address',
            'Jurisdiction', 'EffectiveDate', 'RenewalDate', 'ExpirationDate',
            'ExecutionDate', 'Autorenew', 'Notification Period in days',
            'NotificationPeriodDays', 'Notification_Buffer_Days', 'TerminationNotice',
            'RenewalNotice', 'PerpetualAgreement', 'PerpetualTerminationNoticeDays',
            'Audit Flag', 'AuditReason', 'Lifecycle Stage', 'Risk Level',
            'Archive Flag', 'ArchiveReason', 'GDPR_Applicable', 'Department',
            'Contract_Owner', 'Contract_CoOwner', 'ContractValue', 'ContractCurrency',
            'ContractSummary', 'InitialTerm', 'RenewalTerm', 'CounterpartyRegion',
            'ContractFamily', 'ParentContractId', 'ContractHierarchy', 'DaysToExpiry',
            'TermsInherited', 'SOWStartDate', 'SOWEndDate', 'ParentAgreementDate'
        ]
        
        self.DATE_OUTPUT_COLUMNS = [
            'EffectiveDate', 'ExpirationDate', 'RenewalDate', 'ExecutionDate',
            'SOWStartDate', 'SOWEndDate', 'ParentAgreementDate'
        ]
    
    def _is_ocr_garbage(self, text: str) -> bool:
        """Check if text contains OCR garbage artifacts or is clearly not a party name."""
        if not text or len(str(text)) < 2:
            return True
        
        text = str(text).strip()
        
        # Check for common OCR garbage patterns
        garbage_patterns = [
            r'#{2,}',           # Multiple hash marks (common OCR artifact)
            r'`{2,}',           # Multiple backticks
            r'\\[A-Za-z]',      # File path-like content  
            r'\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\s+##',  # Date followed by hashes
            r'[^\x00-\x7F]{3,}', # Multiple non-ASCII chars in a row (encoding issues)
            r'\*{3,}',          # Multiple asterisks
            r'_{3,}',           # Multiple underscores
            r'\.{5,}',          # Many dots in a row
            r'\[©\s*\)\s*\(©\]', # OCR brackets garbage
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return True
        
        # Check if text has too many digits/symbols vs letters (likely garbage)
        letters = sum(1 for c in text if c.isalpha())
        digits_symbols = sum(1 for c in text if c.isdigit() or c in '#@$%^&*_+={}[]|\\`')
        if len(text) > 10 and digits_symbols > letters:
            return True
        
        # Check for very long text (OCR grabbed too much)
        if len(text) > 150:
            return True
        
        # NEW: Check for contract language that's clearly not a party name
        # These are phrases that appear in contracts but are NOT company names
        contract_language_phrases = [
            r'\bshall\s+(end|begin|commence|terminate|expire)',
            r'\bsubject\s+to\s+the\b',
            r'\bpursuant\s+to\b',
            r'\bin\s+accordance\s+with\b',
            r'\bnotwithstanding\b',
            r'\bwhereas\b',
            r'\bhereby\b',
            r'\bhereunder\b',
            r'\bherein\b',
            r'\bthe\s+company\b',
            r'\bthe\s+parties\b',
            r'\bthe\s+msa\b',
            r'\bthis\s+(agreement|contract|sow)\b',
            r'\bterms\s+(of\s+this|and\s+conditions)\b',
            r'\beffective\s+date\b',
            r'\bexecution\s+date\b',
            r'\btermination\s+(notice|period|date)\b',
            r'\brenewal\s+(term|period|notice)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\bdistribute\s+their\b',
            r'\bfocus:\s*\w+',  # "Focus: Something" pattern
            r'\bpeople,\s*objects',  # Random list
            r'\benterprise[- ]grade\b',
            r'\bproprietary\s+technology\b',
            r'\bwill\s+terminate\s+upon\b',
        ]
        
        text_lower = text.lower()
        for pattern in contract_language_phrases:
            if re.search(pattern, text_lower):
                return True
        
        # NEW: Check if text starts with lowercase (company names are capitalized)
        if text and text[0].islower() and not text.startswith('e'):  # except for eBay, etc
            return True
        
        # NEW: Check if it looks like a sentence fragment (contains common verbs/prepositions)
        sentence_indicators = [
            'shall ', 'will ', 'may ', 'must ', 'should ',
            'upon ', 'under ', 'with respect to', 'in the event',
            'including ', 'excluding ', 'provided that',
        ]
        for indicator in sentence_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def _parse_date(self, date_str: Any) -> Optional[pd.Timestamp]:
        """Parse various date formats to pandas Timestamp"""
        if date_str in ['', 'Unknown', 'Not specified', 'Not found', None]:
            return None
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return pd.Timestamp(date_str)

        normalized = str(date_str).strip()
        if normalized.startswith('D:') and len(normalized) > 8:
            normalized_pdf = normalized[2:]
            for fmt, length in (('%Y%m%d%H%M%S', 14), ('%Y%m%d', 8)):
                try:
                    candidate = pd.to_datetime(normalized_pdf[:length], format=fmt, errors='coerce')
                    if pd.notna(candidate):
                        return pd.Timestamp(candidate)
                except Exception:
                    continue

        formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
            '%Y', '%m/%d/%y'
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(normalized, format=fmt, errors='coerce')
                if pd.notna(parsed):
                    return pd.Timestamp(parsed)
            except Exception:
                continue
        try:
            parsed = pd.to_datetime(normalized, errors='coerce')
            if pd.notna(parsed):
                return pd.Timestamp(parsed)
        except Exception:
            return None
        return None

    def _parse_notice_to_days(self, value: str) -> Optional[int]:
        if not value or value in ['Unknown', 'Not found']:
            return None
        text = str(value)
        days_match = re.search(r'(\d+)\s*(?:days?|day)', text, re.IGNORECASE)
        if days_match:
            return int(days_match.group(1))
        months_match = re.search(r'(\d+)\s*(?:months?|month)', text, re.IGNORECASE)
        if months_match:
            return int(months_match.group(1)) * 30
        weeks_match = re.search(r'(\d+)\s*(?:weeks?|week)', text, re.IGNORECASE)
        if weeks_match:
            return int(weeks_match.group(1)) * 7
        return None

    def _extract_notice_detail(self, metadata: Dict[str, Any], candidate_fields: List[str]) -> Tuple[str, Optional[int]]:
        for field in candidate_fields:
            value = metadata.get(field)
            if value and value not in ['Unknown', 'Not found']:
                return str(value), self._parse_notice_to_days(str(value))
        return '', None

    def _extract_notice_days(self, metadata: Dict[str, Any]) -> Optional[int]:
        _, days = self._extract_notice_detail(metadata, self.NOTICE_CANDIDATE_FIELDS)
        return days

    def _extract_date_field(self, metadata: Dict[str, Any], candidate_fields: List[str]) -> Optional[pd.Timestamp]:
        for field in candidate_fields:
            parsed = self._parse_date(metadata.get(field))
            if parsed is not None:
                return parsed
        return None

    def _extract_clause_text(self, metadata: Dict[str, Any], candidate_fields: List[str]) -> str:
        text, _ = self._extract_notice_detail(metadata, candidate_fields)
        return text

    def _extract_document_created(self, metadata: Dict[str, Any]) -> Optional[pd.Timestamp]:
        created_fields = [
            'document_created', 'doc_created', 'file_created', 'created_at',
            'creation_date', 'modified_at'
        ]
        for field in created_fields:
            parsed = self._parse_date(metadata.get(field))
            if parsed is not None:
                return parsed
        return None

    def _normalize_yes_no(self, value: Any, default: str = 'Unknown') -> str:
        truthy_strings = {'yes', 'true', '1', 'y'}
        falsy_strings = {'no', 'false', '0', 'n'}
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in truthy_strings:
                return 'Yes'
            if value_lower in falsy_strings:
                return 'No'
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        elif isinstance(value, (int, float)):
            if value == 1:
                return 'Yes'
            if value == 0:
                return 'No'
        return default

    def _resolve_counterparty_region(self, metadata: Dict[str, Any], counterparty_address: str) -> Optional[str]:
        explicit_region = metadata.get('counterparty_region') or metadata.get('region')
        if isinstance(explicit_region, str) and explicit_region.strip():
            return explicit_region
        inferred = self._infer_region_from_address(counterparty_address)
        if inferred:
            return inferred
        folder_hint = (metadata.get('agreement_folder') or metadata.get('folder_name') or '').lower()
        if 'emea' in folder_hint or 'europe' in folder_hint:
            return 'EMEA'
        if any(key in folder_hint for key in ['north america', 'na_', 'na-', 'na ']):
            return 'North America'
        if 'apac' in folder_hint or 'asia' in folder_hint:
            return 'APAC'
        return None

    def _detect_perpetual_info(
        self,
        metadata: Dict[str, Any],
        contract_type: str,
        term_texts: List[str]
    ) -> Tuple[str, str, Optional[int]]:
        term_blob = ' '.join(text for text in term_texts if text)
        additional_text = ' '.join(
            str(metadata.get(field, ''))
            for field in ['contract_description', 'contract_summary', 'contract_notes']
        )
        searchable = ' '.join([contract_type or '', term_blob, additional_text]).lower()
        is_flagged = False
        if any(keyword in searchable for keyword in ['perpetual', 'evergreen', 'no fixed term']):
            is_flagged = True
        if self._normalize_yes_no(metadata.get('is_perpetual'), 'No') == 'Yes':
            is_flagged = True
        # Check the 'perpetual' field from enhanced term extraction
        if self._normalize_yes_no(metadata.get('perpetual'), 'No') == 'Yes':
            is_flagged = True
        perpetual_notice_text = self._extract_clause_text(metadata, self.PERPETUAL_NOTICE_FIELDS)
        notice_days = self._parse_notice_to_days(perpetual_notice_text)
        return ('Yes' if is_flagged else 'No', perpetual_notice_text, notice_days)

    def _get_notification_buffer(self, metadata: Dict[str, Any]) -> int:
        buffer_candidate = metadata.get('notification_buffer_days') or metadata.get('notice_buffer_days')
        if isinstance(buffer_candidate, (int, float)):
            return int(buffer_candidate)
        if isinstance(buffer_candidate, str):
            parsed = self._parse_notice_to_days(buffer_candidate)
            if parsed is not None:
                return parsed
        return 30

    def _resolve_notification_days(
        self,
        metadata: Dict[str, Any],
        contract_type: str,
        fallbacks: List[Optional[int]]
    ) -> Optional[int]:
        candidate_fields = [
            'notification_period_days',
            'notification_period_in_days',
            'notice_days',
            'notice_period_days',
            'notice_period'
        ]
        for field in candidate_fields:
            value = metadata.get(field)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                parsed = self._parse_notice_to_days(value)
                if parsed is not None:
                    return parsed
        for fallback in fallbacks:
            if fallback is not None:
                return fallback
        return self._default_notice_days_for_type(contract_type)

    def _default_notice_days_for_type(self, contract_type: str) -> int:
        lower_type = (contract_type or '').lower()
        if 'nda' in lower_type:
            return 30
        if 'master' in lower_type or 'msa' in lower_type:
            return 90
        if 'reseller' in lower_type or 'distributor' in lower_type:
            return 60
        return 30

    def _determine_gdpr(self, metadata: Dict[str, Any], counterparty_address: str, region: Optional[str]) -> Tuple[str, str]:
        gdpr_flag = str(metadata.get('gdpr_applicable', 'No')).title()
        gdpr_reason = metadata.get('gdpr_reason', '')
        # Also check subject_to_gdpr field (used in some ingestion flows)
        if gdpr_flag == 'No':
            subject_to_gdpr = str(metadata.get('subject_to_gdpr', 'No')).lower()
            if subject_to_gdpr in {'yes', 'true', '1'}:
                gdpr_flag = 'Yes'
                gdpr_reason = gdpr_reason or metadata.get('gdpr_reason', 'Subject to GDPR per extraction')
        if gdpr_flag == 'No' and region and region.lower() == 'emea':
            gdpr_flag = 'Yes'
            gdpr_reason = gdpr_reason or 'Counterparty region in EMEA'
        if gdpr_flag == 'No' and counterparty_address:
            address_lower = counterparty_address.lower()
            if any(keyword in address_lower for keyword in self.EMEA_KEYWORDS):
                gdpr_flag = 'Yes'
                gdpr_reason = gdpr_reason or 'Counterparty address in EMEA'
        return gdpr_flag, gdpr_reason

    def _determine_audit_flag(self, effective_date: Optional[pd.Timestamp]) -> Tuple[str, str]:
        if effective_date is None:
            return 'No', ''
        if (pd.Timestamp.now() - effective_date).days >= 730:
            return 'Yes', 'Effective date older than 2 years'
        return 'No', ''

    def _calculate_folder_actions(
        self,
        metadata: Dict[str, Any],
        doc_created: Optional[pd.Timestamp],
        lifecycle_stage: str,
        archive_flag_default: bool
    ) -> Dict[str, Any]:
        folder_notes = metadata.get('document_status_label', '')
        folder_name = (metadata.get('agreement_folder') or metadata.get('folder_name') or '').lower()
        is_draft = metadata.get('is_draft', False) or 'draft' in folder_name
        is_terminated_folder = 'terminated' in folder_name or metadata.get('contract_status', '').lower() == 'terminated'
        archive_flag = archive_flag_default
        archive_reason_parts: List[str] = []
        review_flag = ''

        if is_draft:
            if doc_created is None:
                review_flag = 'Draft folder (age unknown)'
            else:
                now = pd.Timestamp.now(tz=doc_created.tz) if doc_created.tz else pd.Timestamp.now()
                age_days = (now - doc_created).days
                if age_days > 90:
                    archive_flag = True
                    archive_reason_parts.append('Draft folder older than 90 days')
                else:
                    review_flag = f'Draft folder {age_days} days old'

        if is_terminated_folder:
            archive_flag = True
            archive_reason_parts.append('Agreement folder marked terminated')

        lifecycle_override = lifecycle_stage
        if archive_flag and lifecycle_stage not in ['Terminated', 'Archive']:
            lifecycle_override = 'Archive'

        return {
            'archive_flag': archive_flag,
            'archive_reason': '; '.join(dict.fromkeys(part for part in archive_reason_parts if part)),
            'review_flag': review_flag,
            'folder_notes': folder_notes,
            'lifecycle': lifecycle_override
        }

    def _infer_region_from_address(self, address: str) -> Optional[str]:
        if not address:
            return None
        lower_address = address.lower()
        if any(keyword in lower_address for keyword in self.EMEA_KEYWORDS):
            return 'EMEA'
        if any(keyword in lower_address for keyword in ['united states', 'usa', 'canada', 'mexico']):
            return 'North America'
        if any(keyword in lower_address for keyword in ['singapore', 'japan', 'china', 'australia', 'new zealand']):
            return 'APAC'
        return None

    def _fallback_department(self, metadata: Dict[str, Any]) -> str:
        contract_type = (metadata.get('contract_type') or '').lower()
        department_map = {
            'reseller': 'Sales',
            'distributor': 'Sales',
            'channel': 'Sales',
            'partner': 'Sales',
            'nda': 'Legal',
            'non-disclosure': 'Legal',
            'confidentiality': 'Legal',
            'employment': 'HR',
            'contractor': 'HR',
            'consultant': 'HR',
            'marketing': 'Marketing',
            'joint marketing': 'Marketing',
            'procurement': 'Procurement',
            'vendor': 'Procurement',
            'supplier': 'Procurement',
            'oem': 'Engineering',
            'development': 'Engineering',
            'license': 'Legal',
            'software': 'Engineering',
            'master service': 'Operations',
            'msa': 'Operations',
        }
        for keyword, dept in department_map.items():
            if keyword in contract_type:
                return dept
        return 'General'

    def _resolve_department(self, metadata: Dict[str, Any]) -> str:
        return metadata.get('sharepoint_department') or self._fallback_department(metadata)

    def _resolve_department_and_owner(
        self,
        metadata: Dict[str, Any],
        contract_type: str,
        region: Optional[str]
    ) -> Tuple[str, str, str]:
        department = self._resolve_department(metadata)
        owner = metadata.get('sharepoint_owner_primary') or self.default_owner
        co_owner = metadata.get('sharepoint_owner_secondary') or self.default_co_owner
        normalized_region = (region or '').lower()
        contract_lower = (contract_type or '').lower()

        if 'distributor' in contract_lower and normalized_region in self.NA_APAC_REGION_KEYS:
            department = 'NA APAC Distributor Management'
            owner = self.distribution_owner or owner

        if not owner:
            owner = self.default_owner
        if not co_owner:
            co_owner = self.default_co_owner if self.default_co_owner != owner else ''
        elif co_owner == owner and self.default_co_owner != owner:
            co_owner = self.default_co_owner

        return department, owner or '', co_owner or ''

    # =========================================================================
    # CONTRACT FAMILY / HIERARCHY METHODS
    # =========================================================================
    
    def _parse_sow_number_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract SOW number from filename.
        
        Patterns matched:
        - "SOW #10" → 10
        - "SOW#10" → 10
        - "SOW 10" → 10
        - "SOW10" → 10
        - "SOW11" → 11
        - "Statement of Work 5" → 5
        
        Returns: The SOW number if found, None otherwise.
        """
        if not filename:
            return None
        
        filename_lower = filename.lower()
        
        # Try various SOW number patterns
        patterns = [
            r'\bsow\s*#?\s*(\d+)',           # SOW #10, SOW#10, SOW 10, SOW10
            r'statement\s+of\s+work\s*#?\s*(\d+)',  # Statement of Work 5
            r'\bsow(\d+)\b',                 # sow11 (no space)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return int(match.group(1))
        
        return None

    def _parse_amendment_number_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract Amendment number from filename.
        
        Patterns matched:
        - "Amendment 1" → 1
        - "Amendment #1" → 1
        - "Addendum 2" → 2
        - "A1" → 1 (less reliable, only if clearly amendment context)
        
        Returns: The Amendment number if found, None otherwise.
        """
        if not filename:
            return None
        
        filename_lower = filename.lower()
        
        patterns = [
            r'\bamendment\s*#?\s*(\d+)',
            r'\baddendum\s*#?\s*(\d+)',
            r'\bmodification\s*#?\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return int(match.group(1))
        
        return None

    def _parse_schedule_number_from_filename(self, filename: str) -> Optional[int]:
        """Extract Schedule/Exhibit number from filename."""
        if not filename:
            return None
        filename_lower = filename.lower()
        patterns = [
            r'\bschedule\s*#?\s*(\d+)',
            r'\bschedule(\d+)\b',
            r'\bsched\s*#?\s*(\d+)',
            r'\bexhibit\s*#?\s*(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return int(match.group(1))
        return None

    def _determine_contract_hierarchy(self, metadata: Dict[str, Any], contract_type: str, filename: str) -> str:
        """
        Classify document in contract hierarchy:
        - MSA: Master Service Agreement, Master Agreement
        - SOW: Statement of Work, Work Order, Project Order
        - Amendment: Amendment, Addendum, Modification
        - NDA: Non-Disclosure, Confidentiality Agreement
        - Standalone: Independent agreements (Reseller, Distributor, etc.)
        
        IMPORTANT: Filename is checked FIRST because it's typically more reliable
        than contract_type which can be misdetected when a SOW references its parent MSA.
        """
        type_lower = (contract_type or '').lower()
        filename_lower = (filename or '').lower()
        
        # PRIORITY 1: Check FILENAME for specific indicators (most reliable)
        # Check for SOW indicators in filename FIRST
        # Include patterns like "SOW11", "SOW#10", "SOW 5", etc.
        sow_filename_patterns = [
            r'\bsow\s*#?\d+',      # sow#10, sow 5, sow10
            r'\bsow\b',            # standalone "sow"
            r'statement\s+of\s+work',
            r'work\s+order',
            r'project\s+order',
            r'task\s+order',
        ]
        for pattern in sow_filename_patterns:
            if re.search(pattern, filename_lower):
                return 'SOW'
        
        # Check for Amendment indicators in filename
        amendment_keywords = ['amendment', 'addendum', 'modification', 'change order', 'supplement']
        if any(kw in filename_lower for kw in amendment_keywords):
            return 'Amendment'
        
        # Check for NDA in filename
        nda_keywords = ['nda', 'non-disclosure', 'confidentiality', 'mutual nda']
        if any(kw in filename_lower for kw in nda_keywords):
            return 'NDA'
        
        # Check for MSA indicators in filename
        msa_keywords = ['master service', 'master agreement', ' msa ', '_msa_', '-msa-', '_msa.', 
                       'framework agreement', 'umbrella agreement']
        if any(kw in filename_lower for kw in msa_keywords):
            return 'MSA'
        
        # Check for schedules/exhibits in filename
        schedule_keywords = ['schedule', 'exhibit', 'appendix', 'annex', 'attachment']
        if any(kw in filename_lower for kw in schedule_keywords):
            return 'Schedule'
        
        # PRIORITY 2: Check contract_type field (may be misdetected)
        # Only use if filename didn't give us a clear answer
        
        # SOW in contract_type
        if any(kw in type_lower for kw in ['statement of work', 'sow', 'work order', 'project order']):
            return 'SOW'
        
        # Amendment in contract_type
        if any(kw in type_lower for kw in ['amendment', 'addendum', 'modification']):
            return 'Amendment'
        
        # NDA in contract_type
        if any(kw in type_lower for kw in ['nda', 'non-disclosure', 'confidentiality']):
            return 'NDA'
        
        # MSA in contract_type (only if not already matched as something else)
        if any(kw in type_lower for kw in ['master service', 'master agreement', 'msa', 'framework']):
            return 'MSA'
        
        # Default to Standalone for independent agreements
        return 'Standalone'

    def _get_clean_counterparty_name(
        self,
        metadata: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Extract a clean counterparty name with multiple fallbacks.
        
        Priority:
        1. counterparty_name field (if not OCR garbage)
        2. party_counterparty field (if not OCR garbage)
        3. Folder name (parent_folder_name, company_folder)
        4. Extract from filename (before common keywords)
        5. 'Unknown' as last resort
        """
        # Try primary counterparty fields
        for field in ['counterparty_name', 'party_counterparty', 'party_counterparty_name']:
            value = metadata.get(field, '')
            if value and not self._is_ocr_garbage(str(value)):
                return str(value).strip()
        
        # Try folder names (usually reliable)
        for field in ['parent_folder_name', 'doc_parent_folder_name', 'company_folder']:
            value = metadata.get(field, '')
            if value and not self._is_ocr_garbage(str(value)):
                return str(value).strip()
        
        # Try to extract from filename
        # Common patterns: "CompanyName - Agreement.pdf", "CompanyName_NDA.pdf"
        if filename:
            # Remove file extension
            name_part = re.sub(r'\.(pdf|docx?|xlsx?)$', '', filename, flags=re.IGNORECASE)
            # Split on common separators and take first meaningful part
            parts = re.split(r'\s*[-_]\s*|\s+(?:Agreement|Contract|NDA|MSA|SOW)\b', name_part, flags=re.IGNORECASE)
            if parts and parts[0]:
                candidate = parts[0].strip()
                # Check if this looks like a company name (has letters, reasonable length)
                if len(candidate) >= 3 and not self._is_ocr_garbage(candidate):
                    return candidate
        
        return 'Unknown'

    def _generate_contract_family_id(
        self, 
        counterparty_name: str, 
        metadata: Dict[str, Any],
        effective_date: Optional[pd.Timestamp]
    ) -> str:
        """
        Generate a family identifier for grouping related contracts.
        Format: CounterpartyName-YYYY (based on earliest agreement year)
        """
        # Ensure we have a clean counterparty name
        if not counterparty_name or counterparty_name == 'Unknown' or self._is_ocr_garbage(counterparty_name):
            # Try to get from folder name as fallback
            counterparty_name = (
                metadata.get('parent_folder_name') or 
                metadata.get('doc_parent_folder_name') or 
                metadata.get('company_folder') or 
                'Unknown'
            )
        
        # Clean counterparty name for ID (remove special chars, limit length)
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(counterparty_name))[:20]
        
        # Ensure we have something
        if not clean_name:
            clean_name = 'Unknown'
        
        # Use parent agreement date if available, otherwise effective date
        parent_date_str = metadata.get('parent_agreement_date', '')
        if parent_date_str:
            # Extract year from parent agreement reference
            year_match = re.search(r'(19|20)\d{2}', parent_date_str)
            if year_match:
                return f"{clean_name}-{year_match.group()}"
        
        # Fall back to effective date year
        if effective_date is not None:
            return f"{clean_name}-{effective_date.year}"
        
        # Last resort: use current year
        return f"{clean_name}-{datetime.now().year}"

    def _find_parent_contract_id(
        self,
        metadata: Dict[str, Any],
        hierarchy: str,
        counterparty_name: str,
        all_contracts: List[Dict[str, Any]]
    ) -> str:
        """
        Find the parent contract ID for SOWs and Amendments.
        Uses parent_agreement_date from extraction to match MSA.
        """
        # Only SOWs, Amendments, and Schedules have parents
        if hierarchy not in ['SOW', 'Amendment', 'Schedule']:
            return ''
        
        parent_date = metadata.get('parent_agreement_date', '')
        
        # Search for matching MSA
        for contract in all_contracts:
            contract_metadata = contract.get('metadata', {})
            contract_type = (contract_metadata.get('contract_type') or '').lower()
            contract_counterparty = (contract_metadata.get('counterparty_name', '') or 
                                     contract_metadata.get('party_counterparty_name', '') or
                                     contract_metadata.get('parent_folder_name', '') or 
                                     contract_metadata.get('doc_parent_folder_name', ''))
            
            # Check if this is an MSA for the same counterparty
            if 'master' in contract_type or 'msa' in contract_type:
                if self._counterparty_match(counterparty_name, contract_counterparty):
                    # If we have a parent date reference, try to match it
                    if parent_date:
                        contract_effective = contract_metadata.get('effective_date', '')
                        if parent_date in str(contract_effective):
                            return contract.get('id', '')
                    else:
                        # No specific date reference, return first matching MSA
                        return contract.get('id', '')
        
        return ''

    def _counterparty_match(self, name1: str, name2: str) -> bool:
        """Check if two counterparty names likely refer to the same company."""
        if not name1 or not name2:
            return False
        
        # Normalize names - remove common suffixes and clean
        def normalize(name: str) -> str:
            name = name.lower().strip()
            # Remove common company suffixes
            suffixes = ['inc', 'inc.', 'llc', 'ltd', 'limited', 'corp', 'corporation', 
                       'co', 'company', 'pty', 'gmbh', 'ag', 'sa', 'nv', 'bv']
            words = re.sub(r'[^a-zA-Z0-9\s]', '', name).split()
            words = [w for w in words if w.lower() not in suffixes]
            return ''.join(words)
        
        clean1 = normalize(name1)
        clean2 = normalize(name2)
        
        # Direct match
        if clean1 == clean2:
            return True
        
        # One contains the other (for variations like "Matellio" vs "Matellio LLC")
        if clean1 and clean2:
            if clean1 in clean2 or clean2 in clean1:
                return True
            
            # Check if first word matches (company name typically first)
            words1 = clean1.split() if ' ' in name1 else [clean1]
            words2 = clean2.split() if ' ' in name2 else [clean2]
            if words1 and words2 and words1[0] == words2[0]:
                return True
        
        return False

    def _calculate_days_to_expiry(
        self,
        expiration_date: Optional[pd.Timestamp],
        sow_end_date: Optional[pd.Timestamp]
    ) -> Optional[int]:
        """Calculate days until expiration for alerts."""
        now = pd.Timestamp.now()
        
        # Use SOW end date if available, otherwise expiration date
        target_date = sow_end_date if sow_end_date is not None else expiration_date
        
        if target_date is None:
            return None
        
        days = (target_date - now).days
        return days

    def _resolve_inherited_terms(
        self,
        metadata: Dict[str, Any],
        hierarchy: str,
        parent_id: str,
        all_contracts: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve terms from parent contract when child is missing them.
        
        The MSA is the governing document. Child documents (SOW, Amendment, Schedule)
        inherit governing terms from their parent MSA when not explicitly defined.
        
        Inheritable terms:
        - Jurisdiction / Governing Law (CRITICAL - legal requirement)
        - Termination notice period
        - Renewal terms
        - Auto-renewal provisions
        - Notification periods
        - Confidentiality terms
        
        Returns: (inherited_flag, resolved_terms_dict)
        """
        # Standalone agreements and MSAs don't inherit - they ARE the governing document
        if hierarchy in ['MSA', 'NDA', 'Standalone'] or not parent_id:
            return 'No', {}
        
        # Check what terms this child document is missing
        has_jurisdiction = self._has_valid_jurisdiction(metadata)
        has_termination = bool(
            metadata.get('termination_notice') or 
            metadata.get('termination_notice_period') or
            metadata.get('termination_clause')
        )
        has_renewal = bool(
            metadata.get('renewal_term') or 
            metadata.get('auto_renewal')
        )
        has_notification = bool(
            metadata.get('notification_period') or
            metadata.get('notice_period')
        )
        has_initial_term = bool(metadata.get('initial_term'))
        
        # If child has ALL terms defined, no inheritance needed
        if has_jurisdiction and has_termination and has_renewal and has_notification:
            return 'No', {}
        
        # Find parent contract and inherit missing terms
        inherited_terms: Dict[str, Any] = {}
        for contract in all_contracts:
            if contract.get('id') == parent_id:
                parent_meta = contract.get('metadata', {})
                
                # CRITICAL: Jurisdiction / Governing Law inheritance
                # This is the most important - SOWs operate under the MSA's legal framework
                if not has_jurisdiction:
                    parent_jurisdiction = (
                        parent_meta.get('governing_law') or 
                        parent_meta.get('jurisdiction') or ''
                    )
                    if self._is_valid_jurisdiction(parent_jurisdiction):
                        inherited_terms['jurisdiction'] = parent_jurisdiction
                        inherited_terms['governing_law'] = parent_jurisdiction
                
                # Termination notice inheritance
                if not has_termination:
                    inherited_terms['termination_notice'] = parent_meta.get('termination_notice', '')
                    inherited_terms['termination_notice_period'] = parent_meta.get('termination_notice_period', '')
                    inherited_terms['termination_clause'] = parent_meta.get('termination_clause', '')
                
                # Renewal terms inheritance
                if not has_renewal:
                    inherited_terms['renewal_term'] = parent_meta.get('renewal_term', '')
                    inherited_terms['auto_renewal'] = parent_meta.get('auto_renewal', '')
                
                # Notification period inheritance
                if not has_notification:
                    inherited_terms['notification_period'] = parent_meta.get('notification_period', '')
                    inherited_terms['notice_period'] = parent_meta.get('notice_period', '')
                
                # Initial term (some SOWs reference "per the MSA")
                if not has_initial_term:
                    inherited_terms['initial_term'] = parent_meta.get('initial_term', '')
                
                # Filter out empty values
                inherited_terms = {k: v for k, v in inherited_terms.items() if v}
                
                if inherited_terms:
                    return 'Yes', inherited_terms
                break
        
        return 'No', {}

    def _has_valid_jurisdiction(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata contains a valid jurisdiction (not a region name)."""
        jurisdiction = (
            metadata.get('governing_law') or 
            metadata.get('jurisdiction') or ''
        )
        return self._is_valid_jurisdiction(jurisdiction)

    def _is_valid_jurisdiction(self, jurisdiction: str) -> bool:
        """
        Validate that a jurisdiction is a proper legal jurisdiction, not a region.
        
        Valid: "Colorado", "State of Delaware", "England and Wales", "California"
        Invalid: "North America", "EMEA", "APAC", "Not found", ""
        """
        if not jurisdiction:
            return False
        
        jurisdiction_lower = jurisdiction.lower().strip()
        
        # These are INVALID - they're regions, not jurisdictions
        invalid_jurisdictions = [
            'north america', 'south america', 'latin america', 'americas',
            'emea', 'europe', 'middle east', 'africa',
            'apac', 'asia pacific', 'asia', 'pacific',
            'global', 'worldwide', 'international',
            'not found', 'unknown', 'n/a', 'none', 'tbd', 'to be determined',
            'region', 'territory'
        ]
        
        if jurisdiction_lower in invalid_jurisdictions:
            return False
        
        # Valid jurisdictions are typically US states, countries, or specific legal phrases
        # Common valid patterns:
        valid_indicators = [
            # US States
            'colorado', 'california', 'new york', 'delaware', 'texas', 'florida',
            'massachusetts', 'washington', 'illinois', 'georgia', 'michigan',
            'ohio', 'pennsylvania', 'virginia', 'north carolina', 'arizona',
            'nevada', 'oregon', 'minnesota', 'wisconsin', 'tennessee',
            # Common international
            'england', 'wales', 'scotland', 'ireland', 'germany', 'france',
            'netherlands', 'switzerland', 'singapore', 'hong kong', 'australia',
            'canada', 'ontario', 'british columbia', 'quebec',
            'japan', 'china', 'india', 'brazil', 'mexico',
            # Legal phrases indicating valid jurisdiction
            'state of', 'commonwealth of', 'laws of', 'courts of',
        ]
        
        # Check if it contains a valid indicator
        for indicator in valid_indicators:
            if indicator in jurisdiction_lower:
                return True
        
        # If it's short and doesn't match invalid list, might be a state abbreviation
        # (e.g., "CO", "CA", "NY") - these need the context of other data
        if len(jurisdiction_lower) == 2 and jurisdiction_lower.isalpha():
            return True
        
        # If it's longer than typical region names and doesn't match invalid,
        # it's probably a specific jurisdiction
        if len(jurisdiction_lower) > 4 and jurisdiction_lower not in invalid_jurisdictions:
            return True
        
        return False

    def _resolve_jurisdiction(
        self,
        metadata: Dict[str, Any],
        hierarchy: str,
        inherited_terms: Dict[str, Any],
        counterparty_address: str,
        region: str
    ) -> str:
        """
        Resolve jurisdiction with proper validation and inheritance.
        
        Priority:
        1. Document's own governing_law field (if valid)
        2. Document's own jurisdiction field (if valid)
        3. Inherited jurisdiction from parent MSA (for SOW/Amendment)
        4. Extract from counterparty address
        5. Default based on Spectralink's location (Colorado) for US counterparties
        6. "Requires Review" for unknown cases
        """
        # 1. Check document's own governing_law
        doc_governing_law = metadata.get('governing_law', '')
        if self._is_valid_jurisdiction(doc_governing_law):
            return doc_governing_law
        
        # 2. Check document's own jurisdiction
        doc_jurisdiction = metadata.get('jurisdiction', '')
        if self._is_valid_jurisdiction(doc_jurisdiction):
            return doc_jurisdiction
        
        # 3. Use inherited jurisdiction from parent (for child documents)
        if hierarchy in ['SOW', 'Amendment', 'Schedule']:
            inherited_jurisdiction = inherited_terms.get('jurisdiction', '') or inherited_terms.get('governing_law', '')
            if self._is_valid_jurisdiction(inherited_jurisdiction):
                return f"{inherited_jurisdiction} (from MSA)"
        
        # 4. Try to extract from counterparty address
        if counterparty_address:
            extracted = self._extract_jurisdiction_from_address(counterparty_address)
            if extracted:
                return extracted
        
        # 5. For US counterparties, default to Colorado (Spectralink HQ)
        if region == 'North America':
            # Check if address suggests a specific US state
            if counterparty_address:
                state = self._extract_us_state_from_address(counterparty_address)
                if state:
                    return state
            # Default to Colorado for US contracts without clear jurisdiction
            return "Colorado (default - requires verification)"
        
        # 6. Unknown - flag for review
        return "Requires Review"

    def _extract_jurisdiction_from_address(self, address: str) -> str:
        """Extract legal jurisdiction from address if possible."""
        address_lower = address.lower()
        
        # Common country to jurisdiction mappings
        country_jurisdictions = {
            'england': 'England and Wales',
            'united kingdom': 'England and Wales',
            'uk': 'England and Wales',
            'germany': 'Germany',
            'deutschland': 'Germany',
            'france': 'France',
            'netherlands': 'Netherlands',
            'australia': 'Australia',
            'singapore': 'Singapore',
            'hong kong': 'Hong Kong',
            'japan': 'Japan',
            'canada': 'Ontario, Canada',  # Default to Ontario if province not specified
        }
        
        for country, jurisdiction in country_jurisdictions.items():
            if country in address_lower:
                return jurisdiction
        
        return ''

    def _extract_us_state_from_address(self, address: str) -> str:
        """Extract US state from address."""
        # Common US state patterns in addresses
        us_state_patterns = {
            r'\bcolorado\b|\bco\s+\d{5}': 'Colorado',
            r'\bcalifornia\b|\bca\s+\d{5}': 'California',
            r'\bnew york\b|\bny\s+\d{5}': 'New York',
            r'\btexas\b|\btx\s+\d{5}': 'Texas',
            r'\bflorida\b|\bfl\s+\d{5}': 'Florida',
            r'\bmichigan\b|\bmi\s+\d{5}': 'Michigan',
            r'\billinois\b|\bil\s+\d{5}': 'Illinois',
            r'\bgeorgia\b|\bga\s+\d{5}': 'Georgia',
            r'\bdelaware\b|\bde\s+\d{5}': 'Delaware',
            r'\bmassachusetts\b|\bma\s+\d{5}': 'Massachusetts',
            r'\bwashington\b|\bwa\s+\d{5}': 'Washington',
            r'\bvirginia\b|\bva\s+\d{5}': 'Virginia',
            r'\bnorth carolina\b|\bnc\s+\d{5}': 'North Carolina',
            r'\barizona\b|\baz\s+\d{5}': 'Arizona',
            r'\bnevada\b|\bnv\s+\d{5}': 'Nevada',
            r'\boregon\b|\bor\s+\d{5}': 'Oregon',
            r'\bminnesota\b|\bmn\s+\d{5}': 'Minnesota',
            r'\bwisconsin\b|\bwi\s+\d{5}': 'Wisconsin',
            r'\btennessee\b|\btn\s+\d{5}': 'Tennessee',
            r'\bohio\b|\boh\s+\d{5}': 'Ohio',
            r'\bpennsylvania\b|\bpa\s+\d{5}': 'Pennsylvania',
        }
        
        address_lower = address.lower()
        for pattern, state in us_state_patterns.items():
            if re.search(pattern, address_lower):
                return state
        
        return ''

    # =========================================================================
    # CONTRACT ID GENERATION - SLK-[TYPE]-[YYMM]-[SEQ]
    # =========================================================================
    
    # Type code mapping for Contract IDs
    CONTRACT_TYPE_CODES = {
        # MSA - Master Service Agreements only
        'master service agreement': 'MSA',
        'master service agreement (msa)': 'MSA',
        'master agreement': 'MSA',
        'msa': 'MSA',
        
        # SVC - General Service Agreements
        'service agreement': 'SVC',
        'consulting agreement': 'SVC',
        'professional services agreement': 'SVC',
        
        # DST - Distributor Agreements
        'distributor agreement': 'DST',
        'distribution agreement': 'DST',
        
        # RSL - Reseller Agreements
        'reseller agreement': 'RSL',
        'var agreement': 'RSL',
        'value added reseller': 'RSL',
        'partner program agreement': 'RSL',
        
        # OEM - OEM and ODM Agreements
        'oem agreement': 'OEM',
        'odm agreement': 'OEM',
        'original equipment manufacturer': 'OEM',
        'original design manufacturing': 'OEM',
        
        # PUR - Purchase Agreements
        'purchase agreement': 'PUR',
        'purchase order agreement': 'PUR',
        
        # DST - Direct Sales mapped to Distributor (sales channel agreements)
        'sales agreement': 'DST',
        'direct sales agreement': 'DST',
        
        # VND - Vendor/Supplier Agreements
        'vendor/supplier agreement': 'VND',
        'vendor agreement': 'VND',
        'supplier agreement': 'VND',
        
        # DEV - Development Agreements
        'development agreement': 'DEV',
        
        # NDA - All confidentiality agreements
        'non-disclosure agreement (nda)': 'NDA',
        'nda': 'NDA',
        'confidentiality agreement': 'NDA',
        'mutual nda': 'NDA',
        'cda': 'NDA',
        
        # SOW - Statement of Work
        'statement of work (sow)': 'SOW',
        'sow': 'SOW',
        'work order': 'SOW',
        'project order': 'SOW',
        'task order': 'SOW',
        
        # AMD - Amendment (will become A## when linked to parent)
        'amendment/addendum': 'AMD',
        'amendment': 'AMD',
        'addendum': 'AMD',
        'modification': 'AMD',
        'change order': 'AMD',
        
        # Other types
        'termination agreement': 'TRM',
        'correspondence/notice': 'COR',
        'schedule': 'SCH',
        'standalone': 'MSA',  # Default standalone to MSA
    }
    
    # Full names for ContractTitle field
    CONTRACT_TYPE_FULL_NAMES = {
        'MSA': 'Master Service Agreement',
        'SVC': 'Service Agreement',
        'DST': 'Distributor Agreement',
        'RSL': 'Reseller Agreement',
        'OEM': 'OEM Agreement',
        'PUR': 'Purchase Agreement',
        'VND': 'Vendor Agreement',
        'DEV': 'Development Agreement',
        'NDA': 'Non-Disclosure Agreement',
        'SOW': 'Statement of Work',
        'AMD': 'Amendment',
        'TRM': 'Termination Agreement',
        'COR': 'Correspondence',
        'SCH': 'Schedule',
    }
    
    # Complete SharePoint export schema - EXACT columns in EXACT order
    STRICT_SHAREPOINT_SCHEMA = [
        # SYSTEM/IDENTITY
        'Name',                           # Document filename (SharePoint system field)
        'Content Type',                   # SharePoint content type
        'ContractId',                     # The Key (SLK-...)
        'ContractTitle',                  # Full type name or ContractType if no title
        'ContractDuration',               # "MM/DD/YYYY - MM/DD/YYYY"
        
        # PARTIES
        'Party1 Name',                    # Counterparty Name
        'Party1 Address',                 # Counterparty Address
        'Party2 Name',                    # Spectralink entity
        'Party2 Address',                 # Spectralink Address (Louisville or Boulder)
        
        # JURISDICTION
        'Jurisdiction',                   # Governing law
        
        # DATES (Format: YYYY-MM-DD only, NO TIME)
        'EffectiveDate',
        'RenewalDate',
        'ExpirationDate',
        'ExecutionDate',
        
        # RENEWAL/TERMINATION TERMS
        'Autorenew',                      # Yes/No
        'Notification Period in days',    # Notice period
        'NotificationPeriodDays',         # Duplicate for compatibility
        'Notification_Buffer_Days',       # Buffer days
        'TerminationNotice',              # Termination notice text
        'RenewalNotice',                  # Renewal notice text
        'PerpetualAgreement',             # Yes/No
        'PerpetualTerminationNoticeDays', # Renamed from PerpetualNoticeDays
        
        # FLAGS & COMPLIANCE
        'Audit Flag',                     # Yes/No
        'AuditReason',                    # Why flagged
        'Lifecycle Stage',                # Active, Review, Archive, etc.
        'Risk Level',                     # Low/Medium/High (no trailing space)
        'Archive Flag',                   # Yes/No
        'ArchiveReason',                  # Why archived
        'GDPR_Applicable',                # Yes/No
        'Department',                     # Business department
        
        # OWNERSHIP & VALUE
        'Contract_Owner',                 # Email address
        'Contract_CoOwner',               # Email address
        'ContractValue',                  # Numeric only (e.g., 500)
        'ContractCurrency',               # USD, GBP, EUR, DKK
        'ContractSummary',                # Summary/notes
        'InitialTerm',                    # Initial term text
        'RenewalTerm',                    # Renewal term text
        
        # HIERARCHY & FAMILY
        'CounterpartyRegion',             # North America, EMEA, APAC
        'ContractFamily',                 # Grouping (e.g., Matellio-2015)
        'ParentContractId',               # Parent key for SOW/Amendment
        'ContractHierarchy',              # MSA, SOW, Amendment, NDA, Standalone
        'DaysToExpiry',                   # Calculated days
        'TermsInherited',                 # Yes/No
        'SOWStartDate',                   # SOW start date
        'SOWEndDate',                     # SOW end date
        'ParentAgreementDate',            # Parent MSA date
    ]
    
    # Spectralink addresses by date/region
    SPECTRALINK_ADDRESS_NEW = "305 S. Arthur Ave. Louisville, CO 80027"  # After Dec 2024
    SPECTRALINK_ADDRESS_OLD = "2560 55th Street, Boulder, Colorado 80301"  # Before Dec 2024
    SPECTRALINK_ADDRESS_EMEA = "Bygholm Soepark 21 E, 8700 Horsens, Denmark"
    SPECTRALINK_ADDRESS_CUTOFF = pd.Timestamp('2024-12-01')
    
    # Legacy threshold: Before this date, append -LEG
    LEGACY_THRESHOLD = pd.Timestamp('2025-01-01')

    def normalize_party2_name(self, raw_name: Any) -> str:
        """Normalize Spectralink entity labels for Party2."""
        if raw_name in (None, ''):
            return ''
        value = str(raw_name).strip()
        lower = value.lower()
        if any(term in lower for term in ['spectralink europe aps', 'spectralink gmbh', 'spectralink uk ltd']):
            return 'Spectralink Europe ApS'
        if 'spectralink' in lower:
            return 'Spectralink Corporation'
        return value

    def map_party2_address(self, normalized_name: str, execution_date: Any) -> str:
        """Map Spectralink entity to the proper address using execution date."""
        if normalized_name == 'Spectralink Europe ApS':
            return self.SPECTRALINK_ADDRESS_EMEA
        if normalized_name == 'Spectralink Corporation':
            exec_ts = pd.to_datetime(execution_date, errors='coerce') if execution_date else None
            if pd.notna(exec_ts) and exec_ts >= self.SPECTRALINK_ADDRESS_CUTOFF:
                return self.SPECTRALINK_ADDRESS_NEW
            return self.SPECTRALINK_ADDRESS_OLD
        return ''

    def clean_jurisdiction(self, value: Any) -> str:
        """Clean jurisdiction labels per SharePoint expectations."""
        if value in (None, '') or (isinstance(value, float) and pd.isna(value)):
            return 'Requires Review'
        text = str(value).strip()
        if not text:
            return 'Requires Review'
        lower = text.lower()
        if 'default - requires verification' in lower:
            cleaned = text.split('(')[0].strip()
            return cleaned or 'Requires Review'
        return text

    def simplify_contract_title(self, title: Any, hierarchy: Any) -> str:
        """Reduce contract titles to agreement type, falling back to hierarchy mapping."""
        text = str(title or '').strip()
        if ' - ' in text:
            text = text.split(' - ', 1)[0].strip()
        hierarchy_key = str(hierarchy or '').strip().upper()
        hierarchy_map = {
            'MSA': 'Master Service Agreement',
            'NDA': 'Non-Disclosure Agreement',
            'SOW': 'Statement of Work',
            'SCHEDULE': 'Schedule',
            'SCH': 'Schedule',
            'AMENDMENT': 'Amendment',
            'STANDALONE': 'Service Agreement',
        }
        if hierarchy_key in hierarchy_map:
            return hierarchy_map[hierarchy_key]
        return text

    def _standardize_spectralink_entity(self, name: str, region: str) -> str:
        """Normalize Spectralink entity names to a controlled list."""
        if not name:
            return 'Spectralink Corporation'
        lower = name.lower()
        if 'europe' in lower or 'aps' in lower or (region and 'emea' in region.lower()):
            return 'Spectralink Europe ApS'
        if 'gmbh' in lower or 'germany' in lower:
            return 'Spectralink GmbH'
        if 'uk' in lower and 'spectralink' in lower:
            return 'Spectralink UK Ltd'
        if 'spectralink' in lower:
            return 'Spectralink Corporation'
        return 'Spectralink Corporation'

    def _select_spectralink_address_for_entity(
        self,
        entity_name: str,
        effective_date: Optional[pd.Timestamp],
        region: str
    ) -> str:
        """Pick the best Spectralink address based on entity, region, and date."""
        entity_lower = (entity_name or '').lower()
        if 'europe' in entity_lower or 'aps' in entity_lower or (region and 'emea' in region.lower()):
            return self.SPECTRALINK_ADDRESS_EMEA
        if 'gmbh' in entity_lower or 'uk' in entity_lower:
            return self.SPECTRALINK_ADDRESS_EMEA
        eff_date = effective_date
        if eff_date is None:
            eff_date = self._parse_date(effective_date)
        if eff_date is not None and eff_date >= self.SPECTRALINK_ADDRESS_CUTOFF:
            return self.SPECTRALINK_ADDRESS_NEW
        return self.SPECTRALINK_ADDRESS_OLD

    def _normalize_party2_fields(
        self,
        metadata: Dict[str, Any],
        counterparty_name: str,
        effective_date: Optional[pd.Timestamp],
        region: str
    ) -> Tuple[str, str]:
        """Return normalized Spectralink party name/address for SharePoint."""
        name_candidates: List[Optional[str]] = [
            metadata.get('spectralink_name'),
            metadata.get('primary_party_name'),
            metadata.get('party2_name'),
            metadata.get('party1_name') if metadata.get('party1_name') and 'spectralink' in str(metadata.get('party1_name')).lower() else None,
            counterparty_name if counterparty_name and 'spectralink' in counterparty_name.lower() else None,
        ]
        spectralink_name = next((str(c).strip() for c in name_candidates if c), 'Spectralink Corporation')
        spectralink_name = self._standardize_spectralink_entity(spectralink_name, region)

        address_candidates: List[Optional[str]] = [
            metadata.get('spectralink_address'),
            metadata.get('primary_party_address'),
            metadata.get('party2_address'),
        ]
        spectralink_address = next((str(c).strip() for c in address_candidates if c), '')
        if not spectralink_address:
            spectralink_address = self._select_spectralink_address_for_entity(
                spectralink_name,
                effective_date,
                region
            )
        return spectralink_name, spectralink_address

    def _clean_contract_title_text(self, title: str) -> str:
        """Reduce noisy filename artifacts in contract titles."""
        cleaned = title.strip()
        cleaned = re.sub(r'\.(pdf|docx?|xlsx?)$', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?i)\b(fully\s+executed|final|final\s+clean|executed copy)\b', '', cleaned)
        cleaned = re.sub(r'[_\-]+', ' ', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return cleaned.strip(' -_/')

    def _simplify_contract_title(
        self,
        metadata: Dict[str, Any],
        contract_hierarchy: str,
        contract_type: str,
        counterparty_name: str,
        spectralink_name: str
    ) -> str:
        """Generate a concise title aligned with SharePoint expectations."""
        title_candidates = [
            metadata.get('contract_title'),
            metadata.get('document_title'),
            metadata.get('document_name'),
        ]
        for candidate in title_candidates:
            if candidate:
                cleaned = self._clean_contract_title_text(str(candidate))
                if len(cleaned) >= 6:
                    return cleaned
        type_code = self._get_type_code(contract_type, contract_hierarchy)
        full_name = self.CONTRACT_TYPE_FULL_NAMES.get(
            type_code,
            contract_hierarchy or contract_type or 'Contract'
        )
        counterparty_fragment = counterparty_name if counterparty_name not in ['', 'Unknown'] else spectralink_name
        return f"{full_name} - {counterparty_fragment}"

    def _extract_jurisdiction_value(
        self,
        metadata: Dict[str, Any],
        contract_hierarchy: str,
        inherited_terms: Dict[str, Any],
        counterparty_address: str,
        region: str
    ) -> str:
        """Leverage multiple metadata fields before falling back to standard resolver."""
        extra_fields = [
            'jurisdiction_clean',
            'jurisdiction_text',
            'jurisdiction_summary',
            'governing_law_clean',
            'governing_law_text',
            'legal_jurisdiction',
            'legal_venue'
        ]
        for field in extra_fields:
            value = metadata.get(field)
            if value and self._is_valid_jurisdiction(str(value)):
                return str(value)
        parent_value = inherited_terms.get('jurisdiction') or inherited_terms.get('governing_law')
        if parent_value and self._is_valid_jurisdiction(str(parent_value)):
            return f"{parent_value} (from MSA)"
        return self._resolve_jurisdiction(
            metadata,
            contract_hierarchy,
            inherited_terms,
            counterparty_address,
            region
        )
    
    def _get_type_code(self, contract_type: str, hierarchy: str) -> str:
        """Get 3-letter type code from contract type."""
        type_lower = (contract_type or '').lower().strip()
        hierarchy_lower = (hierarchy or '').lower()
        
        # Check hierarchy first for SOW/Amendment
        if hierarchy_lower == 'sow':
            return 'SOW'
        if hierarchy_lower == 'amendment':
            return 'AMD'
        if hierarchy_lower == 'nda':
            return 'NDA'
        
        # Look up in mapping
        if type_lower in self.CONTRACT_TYPE_CODES:
            return self.CONTRACT_TYPE_CODES[type_lower]
        
        # Partial match
        for key, code in self.CONTRACT_TYPE_CODES.items():
            if key in type_lower or type_lower in key:
                return code
        
        # Default to MSA for unknown (will be flagged for review)
        return 'MSA'
    
    def _format_date_for_id(self, date: Optional[pd.Timestamp]) -> str:
        """Format date as YYMM for Contract ID."""
        if date is None:
            # Use current date if no effective date
            return datetime.now().strftime('%y%m')
        return date.strftime('%y%m')
    
    def _is_legacy(self, effective_date: Optional[pd.Timestamp]) -> bool:
        """Check if contract is legacy (before Jan 1, 2025)."""
        if effective_date is None:
            return True  # Unknown dates treated as legacy
        return effective_date < self.LEGACY_THRESHOLD
    
    def _generate_contract_id(
        self,
        sequence: int,
        type_code: str,
        effective_date: Optional[pd.Timestamp],
        hierarchy: str,
        parent_id: str,
        child_sequence: int = 0,
        is_draft: bool = False
    ) -> Tuple[str, str]:
        """
        Generate Contract ID following SLK-[TYPE]-[YYMM]-[SEQ]-[HIERARCHY]-[STATUS] format.
        
        Returns: (contract_id, notes)
        
        Format: [PREFIX]-[TYPE]-[YYMM]-[SEQ]-[HIERARCHY]-[STATUS]
        - PREFIX: SLK
        - TYPE: NDA, MSA, DST, etc.
        - YYMM: Year/Month from effective date
        - SEQ: 4-digit sequence number
        - HIERARCHY: SOW01, A01, etc. (for child documents)
        - STATUS: LEG (for legacy contracts before 2025)
        
        Examples:
        - MSA: SLK-MSA-2511-0001
        - Legacy MSA: SLK-MSA-1805-0022-LEG
        - SOW with parent: SLK-MSA-1510-0001-SOW08-LEG (LEG at END)
        - Amendment with parent: SLK-MSA-2501-0001-A01
        - SOW without parent: SLK-SOW-1805-0022-LEG (flagged)
        """
        notes: list[str] = []
        
        # Format components
        date_part = self._format_date_for_id(effective_date)
        seq_part = f"{sequence:04d}"
        is_leg = self._is_legacy(effective_date)
        
        # Handle child documents (SOW, Amendment)
        if hierarchy in ['SOW', 'Amendment', 'Schedule'] and parent_id:
            # Use parent ID as base, but strip -LEG if present (will add at end)
            parent_base = parent_id.replace('-LEG', '')
            
            # Format: [ParentID]-SOW## or [ParentID]-A##
            suffix_map = {'SOW': 'SOW', 'Amendment': 'A', 'Schedule': 'SCH'}
            suffix = suffix_map.get(hierarchy, 'SOW')
            child_num = f"{child_sequence:02d}" if child_sequence > 0 else "01"
            
            # Build ID: parent base + hierarchy suffix + LEG at end
            contract_id = f"{parent_base}-{suffix}{child_num}"
            if is_leg:
                contract_id += "-LEG"
            return contract_id, ''
        
        # Handle child without parent (flag for review)
        if hierarchy in ['SOW', 'Amendment', 'Schedule'] and not parent_id:
            notes.append(f"Missing Parent Link - {hierarchy} without governing MSA")
            # Generate standalone ID for now
            base_id = f"SLK-{type_code}-{date_part}-{seq_part}"
            if is_leg:
                base_id += "-LEG"
            return base_id, '; '.join(notes)
        
        # Handle drafts
        if is_draft:
            base_id = f"SLK-{type_code}-{date_part}-{seq_part}"
            draft_suffix = f"-D{child_sequence:02d}" if child_sequence > 0 else "-D01"
            # Draft suffix comes before LEG
            base_id += draft_suffix
            if is_leg:
                base_id += "-LEG"
            return base_id, ''
        
        # Standard contract (MSA, NDA, etc.)
        base_id = f"SLK-{type_code}-{date_part}-{seq_part}"
        if is_leg:
            base_id += "-LEG"
        
        return base_id, '; '.join(notes)
    
    def _find_parent_for_child(
        self,
        counterparty: str,
        child_date: Optional[pd.Timestamp],
        all_contracts_sorted: List[Dict[str, Any]]
    ) -> Tuple[str, int]:
        """
        Find parent MSA for a SOW or Amendment based on counterparty and date.
        
        Logic: Look for an MSA/NDA/Standalone agreement for the same counterparty.
        
        Matching priority:
        1. MSA with same counterparty and date <= child date
        2. Any MSA with same counterparty (fallback for mismatched dates)
        3. NDA or Standalone with same counterparty
        
        IMPORTANT: We should ALWAYS find a parent for SOWs if there's any agreement
        for that counterparty. SOWs don't create new parent IDs - they link to existing ones.
        
        Returns: (parent_contract_id, child_sequence_number)
        """
        if not counterparty:
            return '', 1
        
        # Track potential parents
        best_msa_with_date = ''      # MSA with date <= child date
        best_msa_any = ''            # Any MSA for this counterparty (fallback)
        best_other = ''              # NDA/Standalone fallback
        
        # Look for parent contract
        for contract in all_contracts_sorted:
            # Get hierarchy from the contract dict (set in Phase 1)
            hierarchy = contract.get('hierarchy', '')
            contract_id = contract.get('generated_contract_id', '')
            
            # Skip if not a potential parent (SOWs/Amendments can't be parents)
            if hierarchy not in ['MSA', 'NDA', 'Standalone']:
                continue
            
            # Must have an ID (already processed)
            if not contract_id:
                continue
            
            # Check counterparty match - try multiple sources
            contract_counterparty = contract.get('counterparty', '')
            contract_metadata = contract.get('metadata', {})
            contract_folder = (
                contract_metadata.get('parent_folder_name') or 
                contract_metadata.get('doc_parent_folder_name') or 
                contract_metadata.get('company_folder') or ''
            )
            
            # Try matching against counterparty name OR folder name
            matched = (
                self._counterparty_match(counterparty, contract_counterparty) or
                self._counterparty_match(counterparty, contract_folder)
            )
            
            if not matched:
                continue
            
            # Found a potential parent
            parent_date = contract.get('effective_date')
            
            if hierarchy == 'MSA':
                # Track any MSA as fallback
                if not best_msa_any:
                    best_msa_any = contract_id
                
                # Check if date is appropriate (parent on or before child)
                if child_date is None or parent_date is None or parent_date <= child_date:
                    if not best_msa_with_date:
                        best_msa_with_date = contract_id
            elif not best_other:
                # Track NDA/Standalone as last resort
                best_other = contract_id
        
        # Return in priority order
        if best_msa_with_date:
            return best_msa_with_date, 1
        if best_msa_any:
            return best_msa_any, 1  # Use any MSA even if dates don't match perfectly
        if best_other:
            return best_other, 1
        
        return '', 1

    def _calculate_lifecycle_stage(
        self,
        metadata: Dict[str, Any],
        effective_date: Optional[pd.Timestamp],
        expiration_date: Optional[pd.Timestamp]
    ) -> str:
        now = pd.Timestamp.now()
        status_value = (metadata.get('contract_status') or '').lower()
        if metadata.get('is_terminated') or status_value == 'terminated':
            return 'Terminated'
        if metadata.get('is_draft') or status_value == 'draft':
            return 'Draft'
        if status_value in ['archive', 'superseded']:
            return 'Archive'
        if expiration_date is not None:
            if expiration_date < now:
                return 'Expired'
            if (expiration_date - now).days <= 90:
                return 'Review'
        if effective_date is not None and effective_date > now:
            return 'Pending'
        return 'Active'

    def _calculate_risk_level(
        self,
        metadata: Dict[str, Any],
        effective_date: Optional[pd.Timestamp],
        expiration_date: Optional[pd.Timestamp]
    ) -> str:
        risk_score = 0
        now = pd.Timestamp.now()
        if expiration_date is not None:
            days_until = (expiration_date - now).days
            if days_until < 0:
                risk_score += 40
            elif days_until < 30:
                risk_score += 30
            elif days_until < 90:
                risk_score += 20
        if metadata.get('effective_date', 'Unknown') == 'Unknown':
            risk_score += 10
        if metadata.get('contract_type', 'Unknown') == 'Unknown':
            risk_score += 10
        if str(metadata.get('gdpr_applicable', 'No')).lower() in {'yes', 'true'}:
            risk_score += 10
        if metadata.get('termination_for_convenience', 'Unknown') == 'No':
            risk_score += 15
        if risk_score >= 50:
            return 'High'
        if risk_score >= 25:
            return 'Medium'
        return 'Low'

    def _format_contract_duration(self, effective_date: Optional[pd.Timestamp], 
                                  expiration_date: Optional[pd.Timestamp]) -> str:
        """Format contract duration as 'YYYY-MM-DD - YYYY-MM-DD'"""
        parts: List[str] = []
        if pd.notna(effective_date):
            parts.append(effective_date.strftime('%Y-%m-%d'))
        if pd.notna(expiration_date):
            if parts:
                parts.append(' - ')
            parts.append(expiration_date.strftime('%Y-%m-%d'))
        return ''.join(parts) if parts else ''

    def _format_date_output(self, value: Any) -> str:
        """Convert date-like values to YYYY-MM-DD string."""
        if value in (None, '', 'Unknown'):
            return ''
        if isinstance(value, float) and np.isnan(value):
            return ''
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value).strftime('%Y-%m-%d')
        text = str(value).strip()
        if not text:
            return ''
        parsed = pd.to_datetime(text, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
        return text

    def _normalize_contract_id(self, contract_id: str) -> str:
        if not contract_id:
            return contract_id
        normalized = contract_id.strip().replace('--', '-').replace('_', '-').strip('-')
        has_leg = '-LEG' in normalized
        normalized = normalized.replace('-LEG', '').strip('-')
        if has_leg:
            normalized = normalized.rstrip('-') + '-LEG'
        return normalized

    def _derive_parent_contract_id_from_child(self, contract_id: str, hierarchy: str) -> str:
        if not contract_id or hierarchy not in {'SOW', 'Schedule', 'Amendment'}:
            return ''
        match = re.match(r'^(SLK-[A-Z]{3}-\d{4}-\d{4})-(?:SOW|SCH|A)\d{2}(?:-LEG)?$', contract_id)
        if not match:
            return ''
        base = match.group(1)
        suffix = '-LEG' if contract_id.endswith('-LEG') or '-LEG' in contract_id else ''
        return f"{base}{suffix}"

    def _build_family_parent_map(self, contracts: List[Dict[str, Any]]) -> Dict[str, str]:
        """Derive parent ContractIds for each ContractFamily from already generated IDs."""
        parent_types = {'MSA', 'Standalone', 'NDA'}
        hierarchy_priority = {'MSA': 0, 'Standalone': 1, 'NDA': 2}
        family_candidates: Dict[str, List[Tuple[int, Optional[pd.Timestamp], str]]] = {}

        for contract in contracts:
            family = str(contract.get('contract_family') or '').strip()
            if not family:
                continue
            hierarchy = contract.get('hierarchy', '')
            if hierarchy not in parent_types:
                continue
            parent_id = self._normalize_contract_id(contract.get('generated_contract_id', ''))
            if not parent_id:
                continue
            priority = hierarchy_priority.get(hierarchy, 99)
            effective_date = contract.get('effective_date')
            eff_ts = effective_date if isinstance(effective_date, pd.Timestamp) else None
            if family not in family_candidates:
                family_candidates[family] = []
            family_candidates[family].append((priority, eff_ts, parent_id))

        family_parent_map: Dict[str, str] = {}
        for family, candidates in family_candidates.items():
            candidates.sort(key=lambda item: (
                item[0],
                item[1] if item[1] is not None and pd.notna(item[1]) else pd.Timestamp.max
            ))
            family_parent_map[family] = candidates[0][2]

        return family_parent_map

    def _apply_baseline_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.BASELINE_EXPORT_COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df[self.BASELINE_EXPORT_COLUMNS]
    
    
    def _get_party_reference_name(self, party_name: str, is_spectralink: bool, 
                                 contract_type: str) -> str:
        """Determine party reference name (role in contract)"""
        contract_lower = contract_type.lower()
        
        if is_spectralink:
            # Spectralink's role
            if 'reseller' in contract_lower or 'distributor' in contract_lower:
                return 'Vendor'
            elif 'nda' in contract_lower:
                return 'Discloser'
            elif 'procurement' in contract_lower or 'vendor' in contract_lower:
                return 'Client'
            else:
                return 'Client'
        else:
            # Counterparty's role
            if 'reseller' in contract_lower:
                return 'Reseller'
            elif 'distributor' in contract_lower:
                return 'Distributor'
            elif 'nda' in contract_lower:
                return 'Recipient'
            elif 'procurement' in contract_lower or 'vendor' in contract_lower:
                return 'Vendor'
            elif 'hotel' in contract_lower:
                return 'Hotel'
            elif 'contractor' in contract_lower:
                return 'Contractor'
            else:
                return 'Party'
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export ChromaDB collection to SharePoint-formatted DataFrame"""
        logger.info("🔄 Fetching all contracts from ChromaDB...")
        
        # Get all documents with metadata
        results = self.collection.get(include=['metadatas', 'documents'])
        
        if not results['ids']:
            logger.warning("⚠️  No contracts found in ChromaDB")
            return pd.DataFrame()
        
        logger.info(f"✅ Retrieved {len(results['ids'])} contracts")
        
        # Build records list
        records: List[Dict[str, Any]] = []
        documents: List[str] = list(results.get('documents') or [])
        metadatas: List[Dict[str, Any]] = [dict(m or {}) for m in (results.get('metadatas') or [])]

        # =====================================================================
        # NON-CONTRACT FILTER: Skip supporting documents that aren't actual contracts
        # =====================================================================
        NON_CONTRACT_PATTERNS = [
            r'\bchecklist\b',
            r'\bsummary\b',
            r'\btemplate\b',
            r'\bdraft\s*v?\d*\b',  # "draft", "draft v2", etc. (but NOT contract drafts)
            r'\bnotes?\b',
            r'\breview\b',
            r'\bcomments?\b',
            r'\bredline\b',
            r'\btracking\b',
            r'\bnotification\b',
            r'\bnotice\b',
            r'end[-\s]*of[-\s]*life',
            r'\beol\b',
            r'power\s+of\s+attorney',
            r'\bappendix\b',
            r'\bappendixes\b',
            r'\bappendices\b',
            r'\bconsent\s+letter\b',
        ]
        
        def _is_non_contract_file(filename: str) -> bool:
            """Check if filename indicates a supporting document, not an actual contract."""
            fn_lower = filename.lower()
            for pattern in NON_CONTRACT_PATTERNS:
                if re.search(pattern, fn_lower):
                    return True
            return False

        # =====================================================================
        # PHASE 1: Build contract list with preliminary data for sorting
        # =====================================================================
        all_contracts: List[Dict[str, Any]] = []
        skipped_non_contracts: List[str] = []
        
        for idx, db_id in enumerate(results['ids']):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            
            # Get filename early for filtering
            filename = str(metadata.get('filename') or metadata.get('document_name') or db_id)
            
            # Skip non-contract files (checklists, summaries, etc.)
            if _is_non_contract_file(filename):
                skipped_non_contracts.append(filename)
                continue
            
            # Extract effective date for sorting
            effective_date = self._extract_date_field(metadata, self.EFFECTIVE_DATE_FIELDS)
            
            # Determine hierarchy type
            contract_type = str(metadata.get('contract_type') or 'Unknown')
            filename = str(metadata.get('filename') or metadata.get('document_name') or db_id)
            hierarchy = self._determine_contract_hierarchy(metadata, contract_type, filename)

            # Get the best counterparty value once so every phase uses the same string
            clean_counterparty = self._get_clean_counterparty_name(metadata, filename)
            counterparty = clean_counterparty
            
            contract_family_id = self._generate_contract_family_id(clean_counterparty, metadata, effective_date)
            
            # Check if draft
            is_draft = bool(metadata.get('is_draft')) or 'draft' in (metadata.get('contract_status') or '').lower()
            
            all_contracts.append({
                'db_id': db_id,
                'idx': idx,
                'metadata': metadata,
                'effective_date': effective_date,
                'hierarchy': hierarchy,
                'counterparty': counterparty,
                'counterparty_name': clean_counterparty,
                'contract_type': contract_type,
                'filename': filename,
                'is_draft': is_draft,
                'contract_family': contract_family_id,
                'generated_contract_id': '',  # Will be filled in Phase 2
            })
        
        # Log skipped non-contract files
        if skipped_non_contracts:
            logger.info(f"⏭️  Skipped {len(skipped_non_contracts)} non-contract files (checklists, summaries, etc.):")
            for skipped in skipped_non_contracts:
                logger.info(f"   - {skipped}")
        
        # =====================================================================
        # PHASE 2: Sort by effective date and generate Contract IDs
        # =====================================================================
        # Sort: MSA/NDA/Standalone first (parents), then SOW/Amendments (children)
        # Within each group, sort by effective date
        
        def sort_key(c: Dict[str, Any]) -> Tuple[int, Any]:
            hierarchy_order = {'MSA': 0, 'NDA': 1, 'Standalone': 2, 'SOW': 3, 'Amendment': 4, 'Schedule': 5}
            h_order = hierarchy_order.get(c['hierarchy'], 2)
            date = c['effective_date'] if c['effective_date'] is not None else pd.Timestamp('1900-01-01')
            return (h_order, date)
        
        all_contracts_sorted = sorted(all_contracts, key=sort_key)
        
        # Generate IDs - global sequence counter
        global_sequence = 1
        parent_child_counts: Dict[str, Dict[str, int]] = {}  # {parent_id: {'SOW': count, 'Amendment': count}}
        # Track used SOW/Amendment numbers per parent to avoid duplicates
        parent_child_used_numbers: Dict[str, Dict[str, Set[int]]] = {}
        
        for contract in all_contracts_sorted:
            hierarchy = contract['hierarchy']
            effective_date = contract['effective_date']
            counterparty = contract['counterparty']
            contract_type = contract['contract_type']
            is_draft = contract['is_draft']
            filename = contract['filename']
            
            type_code = self._get_type_code(contract_type, hierarchy)
            
            # For SOW/Amendment, try to find parent and parse number from filename
            parent_id = ''
            child_seq = 1
            
            if hierarchy in ['SOW', 'Amendment', 'Schedule']:
                # Find parent from already-processed contracts
                parent_id, _ = self._find_parent_for_child(
                    counterparty, effective_date, 
                    [c for c in all_contracts_sorted if c['generated_contract_id']]
                )
                
                # Parse SOW/Amendment number from filename
                if hierarchy == 'SOW':
                    parsed_num = self._parse_sow_number_from_filename(filename)
                elif hierarchy == 'Amendment':
                    parsed_num = self._parse_amendment_number_from_filename(filename)
                else:
                    parsed_num = self._parse_schedule_number_from_filename(filename)
                
                if parsed_num is not None:
                    # Use the parsed number from filename
                    child_seq = parsed_num
                elif parent_id:
                    # No number in filename - generate sequential number
                    if parent_id not in parent_child_counts:
                        parent_child_counts[parent_id] = {'SOW': 0, 'Amendment': 0, 'Schedule': 0}
                    parent_child_counts[parent_id][hierarchy] = parent_child_counts[parent_id].get(hierarchy, 0) + 1
                    child_seq = parent_child_counts[parent_id][hierarchy]
                
                # Track used numbers to detect duplicates
                if parent_id:
                    if parent_id not in parent_child_used_numbers:
                        parent_child_used_numbers[parent_id] = {'SOW': set(), 'Amendment': set(), 'Schedule': set()}
                    parent_child_used_numbers[parent_id][hierarchy].add(child_seq)
            
            # Generate the ID
            contract_id, notes = self._generate_contract_id(
                sequence=global_sequence,
                type_code=type_code,
                effective_date=effective_date,
                hierarchy=hierarchy,
                parent_id=parent_id,
                child_sequence=child_seq,
                is_draft=is_draft
            )
            
            contract['generated_contract_id'] = contract_id
            contract['id_notes'] = notes
            
            # Only increment global sequence for standalone/parent contracts
            if hierarchy not in ['SOW', 'Amendment', 'Schedule'] or not parent_id:
                global_sequence += 1
        
        logger.info(f"📝 Generated {global_sequence - 1} unique Contract IDs")
        
        family_parent_map = self._build_family_parent_map(all_contracts_sorted)

        # =====================================================================
        # PHASE 3: Build full records with all metadata
        # =====================================================================
        # Re-sort by original index to maintain ChromaDB order for records
        all_contracts_by_idx = {c['idx']: c for c in all_contracts_sorted}

        for idx, db_id in enumerate(results['ids']):
            # Skip non-contract files that were filtered out in Phase 1
            if idx not in all_contracts_by_idx:
                continue
                
            contract_data = all_contracts_by_idx[idx]
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            raw_document = documents[idx] if idx < len(documents) else ''
            document = str(raw_document or '')
            
            # Use pre-computed values from Phase 1 & 2
            contract_id = self._normalize_contract_id(contract_data.get('generated_contract_id', db_id))
            contract_hierarchy = contract_data.get('hierarchy', 'Standalone')
            execution_date = self._extract_date_field(metadata, self.EXECUTION_DATE_FIELDS)
            effective_date = contract_data.get('effective_date')
            expiration_date = self._extract_date_field(metadata, self.EXPIRATION_DATE_FIELDS)
            termination_date = self._parse_date(metadata.get('termination_date'))
            if expiration_date is None and termination_date is not None:
                expiration_date = termination_date
            renewal_date = self._extract_date_field(metadata, self.RENEWAL_DATE_FIELDS)
            doc_created = self._extract_document_created(metadata)

            contract_type = str(metadata.get('contract_type') or 'Unknown')
            
            # Extract filename early (needed for counterparty fallback)
            filename = str(metadata.get('filename') or metadata.get('document_name') or contract_id)

            # Extract counterparty name with multiple fallbacks and validation
            counterparty_name = contract_data.get('counterparty_name') or self._get_clean_counterparty_name(metadata, filename)

            if counterparty_name and 'spectralink' in counterparty_name.lower():
                fallback_counterparty = str(
                    metadata.get('primary_party_name') or
                    metadata.get('parent_folder_name') or
                    metadata.get('doc_parent_folder_name') or
                    'Unknown'
                )
                if fallback_counterparty and 'spectralink' not in fallback_counterparty.lower():
                    counterparty_name = fallback_counterparty

            counterparty_address = str(
                metadata.get('counterparty_address') or metadata.get('party1_address') or ''
            )

            region = self._resolve_counterparty_region(metadata, counterparty_address)
            spectralink_name, spectralink_address = self._normalize_party2_fields(
                metadata,
                counterparty_name,
                effective_date,
                region or ''
            )
            gdpr_flag, _ = self._determine_gdpr(metadata, counterparty_address, region)
            lifecycle = self._calculate_lifecycle_stage(metadata, effective_date, expiration_date)
            risk_level = self._calculate_risk_level(metadata, effective_date, expiration_date)
            audit_flag, audit_reason = self._determine_audit_flag(effective_date)

            initial_term_text = self._extract_clause_text(metadata, self.INITIAL_TERM_FIELDS)
            renewal_term_text = self._extract_clause_text(metadata, self.RENEWAL_TERM_FIELDS)
            termination_notice_text, termination_notice_days = self._extract_notice_detail(
                metadata, self.TERMINATION_NOTICE_FIELDS
            )
            renewal_notice_text, renewal_notice_days = self._extract_notice_detail(
                metadata, self.RENEWAL_NOTICE_FIELDS
            )
            notification_days = self._resolve_notification_days(
                metadata,
                contract_type,
                [termination_notice_days, renewal_notice_days]
            )
            perpetual_flag, _, perpetual_notice_days = self._detect_perpetual_info(
                metadata,
                contract_type,
                [initial_term_text, renewal_term_text, termination_notice_text, renewal_notice_text]
            )
            notification_buffer_days = self._get_notification_buffer(metadata)

            auto_renewal = self._normalize_yes_no(metadata.get('auto_renewal'), 'Unknown')
            # NOTE: Jurisdiction is resolved AFTER inheritance to allow SOW/Amendment
            # to inherit from parent MSA. See below after inherited_terms is computed.
            
            # filename already extracted above (needed early for counterparty fallback)
            contract_title = self._simplify_contract_title(
                metadata,
                contract_hierarchy,
                contract_type,
                counterparty_name,
                spectralink_name
            )

            department, owner, co_owner = self._resolve_department_and_owner(metadata, contract_type, region)

            # Contract Family / Hierarchy - use pre-computed hierarchy from Phase 2
            # contract_hierarchy already set from contract_data
            contract_family_id = contract_data.get('contract_family') or self._generate_contract_family_id(counterparty_name, metadata, effective_date)
            
            # Find parent for SOW/Amendment - use the generated parent ID from Phase 2
            parent_contract_id = ''
            if contract_hierarchy in ['SOW', 'Amendment', 'Schedule']:
                # Extract parent ID from generated contract ID if it's a child
                # Format: SLK-MSA-YYMM-SEQ-SOW## means parent is SLK-MSA-YYMM-SEQ
                if any(suffix in contract_id for suffix in ['-SOW', '-A0', '-SCH']):
                    parts = contract_id.rsplit('-', 1)
                    if len(parts) == 2 and (
                        parts[1].startswith('SOW') or parts[1].startswith('A') or parts[1].startswith('SCH')
                    ):
                        parent_contract_id = parts[0]
            parent_contract_id = parent_contract_id or self._derive_parent_contract_id_from_child(contract_id, contract_hierarchy)
            if not parent_contract_id and contract_family_id:
                family_parent_id = family_parent_map.get(contract_family_id)
                if family_parent_id and family_parent_id != contract_id:
                    parent_contract_id = family_parent_id
            
            # SOW-specific dates
            sow_start_date = self._parse_date(metadata.get('sow_start_date'))
            sow_end_date = self._parse_date(metadata.get('sow_end_date'))
            parent_agreement_date = self._parse_date(metadata.get('parent_agreement_date'))
            
            # Days to expiry calculation (for Power Automate alerts)
            days_to_expiry = self._calculate_days_to_expiry(expiration_date, sow_end_date)
            
            # Check for term inheritance from parent
            # Build a compatible list for the inheritance function
            all_contracts_for_inheritance = [
                {'id': c['generated_contract_id'], 'metadata': c['metadata']} 
                for c in all_contracts_sorted
            ]
            terms_inherited, inherited_terms = self._resolve_inherited_terms(
                metadata, contract_hierarchy, parent_contract_id, all_contracts_for_inheritance
            )
            
            # Apply inherited terms if available
            if inherited_terms:
                if not termination_notice_text and inherited_terms.get('termination_notice'):
                    termination_notice_text = f"[Inherited from MSA] {inherited_terms['termination_notice']}"
                if not renewal_term_text and inherited_terms.get('renewal_term'):
                    renewal_term_text = f"[Inherited from MSA] {inherited_terms['renewal_term']}"
                if auto_renewal == 'Unknown' and inherited_terms.get('auto_renewal'):
                    auto_renewal = self._normalize_yes_no(inherited_terms['auto_renewal'], 'Unknown')
                if not initial_term_text and inherited_terms.get('initial_term'):
                    initial_term_text = f"[Inherited from MSA] {inherited_terms['initial_term']}"

            # NOW resolve jurisdiction - after inheritance so SOW/Amendment can inherit from MSA
            # This is critical: jurisdiction cannot be a region (North America, EMEA)
            # It must be a specific legal jurisdiction (Colorado, England, etc.)
            jurisdiction = self._extract_jurisdiction_value(
                metadata,
                contract_hierarchy,
                inherited_terms,
                counterparty_address,
                region or ''
            )

            archive_flag_default = lifecycle in ['Archive', 'Terminated', 'Expired']
            folder_actions = self._calculate_folder_actions(
                metadata,
                doc_created,
                lifecycle,
                archive_flag_default
            )

            contract_value_str = ''
            contract_currency = ''

            execution_date_value = execution_date or effective_date
            renewal_date_value = renewal_date if renewal_date is not None else (
                expiration_date if auto_renewal == 'Yes' else None
            )

            record: Dict[str, Any] = {
                'Name': filename,
                'Content Type': 'WRT Global B2B Contract Processing',
                'ContractId': contract_id,
                'ContractTitle': contract_title,
                'ContractDuration': self._format_contract_duration(effective_date, expiration_date),
                'Party1 Name': counterparty_name,
                'Party1 Address': '',  # Leave blank for LLM enrichment later
                'Party2 Name': spectralink_name,
                'Party2 Address': spectralink_address,
                'Jurisdiction': jurisdiction,
                'EffectiveDate': self._format_date_output(effective_date),
                'RenewalDate': self._format_date_output(renewal_date_value),
                'ExpirationDate': self._format_date_output(expiration_date),
                'ExecutionDate': self._format_date_output(execution_date_value),
                'Autorenew': auto_renewal,
                'Notification Period in days': notification_days if notification_days is not None else np.nan,
                'NotificationPeriodDays': notification_days if notification_days is not None else np.nan,
                'Notification_Buffer_Days': notification_buffer_days,
                'TerminationNotice': termination_notice_text,
                'RenewalNotice': renewal_notice_text,
                'PerpetualAgreement': perpetual_flag,
                'PerpetualTerminationNoticeDays': (
                    perpetual_notice_days if perpetual_notice_days is not None else np.nan
                ),
                'Audit Flag': audit_flag,
                'AuditReason': audit_reason,
                'Lifecycle Stage': folder_actions['lifecycle'],
                'Risk Level': risk_level,
                'Archive Flag': 'Yes' if folder_actions['archive_flag'] else 'No',
                'ArchiveReason': folder_actions['archive_reason'],
                'GDPR_Applicable': gdpr_flag,
                'Department': department,
                'Contract_Owner': owner,
                'Contract_CoOwner': co_owner,
                'ContractValue': contract_value_str,
                'ContractCurrency': contract_currency,
                'ContractSummary': document[:500] + '...' if document and len(document) > 500 else document,
                'InitialTerm': initial_term_text,
                'RenewalTerm': renewal_term_text,
                'CounterpartyRegion': region or '',
                'ContractFamily': contract_family_id,
                'ParentContractId': parent_contract_id,
                'ContractHierarchy': contract_hierarchy,
                'DaysToExpiry': days_to_expiry if days_to_expiry is not None else np.nan,
                'TermsInherited': terms_inherited,
                'SOWStartDate': self._format_date_output(sow_start_date),
                'SOWEndDate': self._format_date_output(sow_end_date),
                'ParentAgreementDate': self._format_date_output(parent_agreement_date),
            }

            records.append(record)
        
        df = pd.DataFrame(records)
        if not df.empty:
            df['Party2 Name'] = df['Party2 Name'].apply(self.normalize_party2_name)
            df['Party2 Address'] = df.apply(
                lambda row: self.map_party2_address(row['Party2 Name'], row.get('ExecutionDate')),
                axis=1
            )
            df['Jurisdiction'] = df['Jurisdiction'].apply(self.clean_jurisdiction)
            df['ContractTitle'] = df.apply(
                lambda row: self.simplify_contract_title(row['ContractTitle'], row.get('ContractHierarchy')),
                axis=1
            )
            if 'ContractSummary' in df.columns:
                df['ContractSummary'] = None
        df = self._apply_baseline_schema(df)
        logger.info(f"✅ Built DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def export_to_excel(self, filename: Optional[str] = None) -> Optional[Path]:
        """Export to Excel file formatted for SharePoint import"""
        df = self.export_to_dataframe()
        
        if df.empty:
            logger.warning("⚠️  No data to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"contracts_for_sharepoint_{timestamp}.xlsx"
        
        output_path = self.output_dir / filename
        
        # Export with Excel formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(cast(Any, writer), sheet_name='Contracts', index=False)  # type: ignore[arg-type]
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Contracts']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"✅ Exported to: {output_path}")
        return output_path
    
    def validate_schema(self, sharepoint_example: str = "sharepoint list example.xlsx") -> Dict[str, Any]:
        """Compare our export schema with SharePoint example"""
        logger.info(f"🔍 Validating schema against {sharepoint_example}")
        
        try:
            sp_df = pd.read_excel(sharepoint_example, sheet_name=0)  # type: ignore[arg-type]
            sp_columns = set(sp_df.columns.tolist())
        except Exception as e:
            logger.error(f"❌ Could not read SharePoint example: {e}")
            return {'error': str(e)}
        
        our_columns = set(self.SHAREPOINT_SCHEMA.keys())
        
        result: Dict[str, Any] = {
            'matching': sp_columns & our_columns,
            'missing_from_export': sp_columns - our_columns,
            'extra_in_export': our_columns - sp_columns,
            'total_sharepoint': len(sp_columns),
            'total_export': len(our_columns),
            'match_percentage': len(sp_columns & our_columns) / len(sp_columns) * 100
        }
        
        logger.info(f"📊 Schema Comparison Results:")
        logger.info(f"   SharePoint columns: {result['total_sharepoint']}")
        logger.info(f"   Export columns: {result['total_export']}")
        logger.info(f"   Matching: {len(result['matching'])} ({result['match_percentage']:.1f}%)")
        
        if result['missing_from_export']:
            logger.warning(f"   ⚠️  Missing from export: {result['missing_from_export']}")
        if result['extra_in_export']:
            logger.info(f"   ➕ Extra in export: {result['extra_in_export']}")
        
        return result
    
    def preview(self, n: int = 5) -> pd.DataFrame:
        """Preview first n rows of export"""
        df = self.export_to_dataframe()
        return df.head(n)
    
    def _apply_strict_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame to strict SharePoint schema.
        Maps ALL columns defined in STRICT_SHAREPOINT_SCHEMA with exact rules.
        """
        logger.info("📋 Applying strict SharePoint schema...")
        
        # Helper: Format dates as YYYY-MM-DD only (no time)
        def format_date(val: Any) -> Optional[str]:
            if pd.isna(val) or val is None or val == '':
                return None
            if hasattr(val, 'strftime'):
                return val.strftime('%Y-%m-%d')
            try:
                parsed = pd.to_datetime(val)
                if pd.notna(parsed):
                    return parsed.strftime('%Y-%m-%d')
            except:
                pass
            return None
        
        # Helper: Parse contract value to numeric and extract currency
        def parse_contract_value(val: Any, region: str) -> Tuple[Optional[float], str]:
            """Extract numeric value and currency from contract value string."""
            if pd.isna(val) or val is None or val == '' or val == 'Unknown' or val == 'Not found':
                return None, ''
            
            val_str = str(val).strip()
            
            # Detect currency from symbol
            currency = ''
            if '£' in val_str:
                currency = 'GBP'
            elif '€' in val_str:
                currency = 'EUR'
            elif 'kr' in val_str.lower() or 'dkk' in val_str.lower():
                currency = 'DKK'
            elif '$' in val_str:
                currency = 'USD'
            else:
                # Default based on region
                if region and 'emea' in region.lower():
                    currency = 'EUR'
                else:
                    currency = 'USD'
            
            # Extract numeric value
            # Remove currency symbols and common formatting
            numeric_str = re.sub(r'[£$€]', '', val_str)
            numeric_str = re.sub(r'[,\s]', '', numeric_str)
            numeric_str = re.sub(r'(USD|GBP|EUR|DKK|kr)', '', numeric_str, flags=re.IGNORECASE)
            numeric_str = numeric_str.strip()
            
            try:
                numeric_val = float(numeric_str)
                return numeric_val, currency
            except ValueError:
                return None, currency
        
        # Helper: Get Spectralink address based on date and region
        def get_spectralink_address(effective_date: Any, region: str) -> str:
            """Return appropriate Spectralink address based on contract date and region."""
            if region and 'emea' in region.lower():
                return self.SPECTRALINK_ADDRESS_EMEA
            
            # Parse date to determine old vs new address
            if effective_date:
                try:
                    if hasattr(effective_date, 'timestamp'):
                        eff_ts = effective_date
                    else:
                        eff_ts = pd.to_datetime(effective_date)
                    
                    if pd.notna(eff_ts) and eff_ts >= self.SPECTRALINK_ADDRESS_CUTOFF:
                        return self.SPECTRALINK_ADDRESS_NEW
                except:
                    pass
            
            return self.SPECTRALINK_ADDRESS_OLD
        
        # Build records
        records: List[Dict[str, Any]] = []
        
        for _, row in df.iterrows():
            # Get hierarchy - this determines the contract type for SOW/Amendment
            hierarchy = str(row.get('ContractHierarchy', '') or '')
            contract_id = str(row.get('ContractId', '') or '')
            filename = str(row.get('Name', '') or '')
            region = str(row.get('CounterpartyRegion', '') or '')
            
            # Determine type code based on HIERARCHY first (SOW/Amendment), then ContractId
            if hierarchy == 'SOW':
                type_code = 'SOW'
            elif hierarchy == 'Amendment':
                type_code = 'AMD'
            elif contract_id.startswith('SLK-'):
                parts = contract_id.split('-')
                type_code = parts[1] if len(parts) >= 2 else 'MSA'
            else:
                type_code = 'MSA'
            
            # Get full type name for ContractTitle
            contract_title = self.CONTRACT_TYPE_FULL_NAMES.get(type_code, hierarchy or 'Contract')
            
            # Get dates
            effective_date = row.get('EffectiveDate')
            expiration_date = row.get('ExpirationDate')
            execution_date = row.get('ExecutionDate')
            renewal_date = row.get('RenewalDate')
            sow_start = row.get('SOWStartDate')
            sow_end = row.get('SOWEndDate')
            
            # Contract Duration - format as "MM/DD/YYYY - MM/DD/YYYY"
            contract_duration = ''
            if effective_date and pd.notna(effective_date):
                try:
                    eff_str = effective_date.strftime('%m/%d/%Y') if hasattr(effective_date, 'strftime') else ''
                    if expiration_date and pd.notna(expiration_date):
                        exp_str = expiration_date.strftime('%m/%d/%Y') if hasattr(expiration_date, 'strftime') else ''
                        contract_duration = f"{eff_str} - {exp_str}"
                    else:
                        contract_duration = eff_str
                except:
                    pass
            
            # Parse contract value and currency
            contract_value_raw = row.get('ContractValue', '')
            contract_value_numeric, contract_currency = parse_contract_value(contract_value_raw, region)
            
            # Get Spectralink address based on date and region
            spectralink_address = get_spectralink_address(effective_date, region)
            
            # Ensure ParentContractId is set for SOW/Amendment
            parent_id = str(row.get('ParentContractId', '') or '')
            parent_agreement_date_raw = row.get('ParentAgreementDate', '')
            parent_agreement_date = format_date(parent_agreement_date_raw) or ''
            if hierarchy in ['SOW', 'Amendment', 'Schedule'] and not parent_id:
                parent_id = '[MISSING - Requires Review]'
            
            # Handle PerpetualNoticeDays -> PerpetualTerminationNoticeDays rename
            perpetual_notice_days = row.get('PerpetualNoticeDays')
            if pd.isna(perpetual_notice_days):
                perpetual_notice_days = None
            elif isinstance(perpetual_notice_days, float):
                perpetual_notice_days = int(perpetual_notice_days) if not pd.isna(perpetual_notice_days) else None
            
            # Notification period - ensure integer
            notification_period = row.get('Notification Period in days')
            if pd.notna(notification_period) and isinstance(notification_period, (int, float)):
                notification_period = int(notification_period)
            else:
                notification_period = None
            
            notification_buffer = row.get('Notification_Buffer_Days')
            if pd.notna(notification_buffer) and isinstance(notification_buffer, (int, float)):
                notification_buffer = int(notification_buffer)
            else:
                notification_buffer = None
            
            # Days to expiry - ensure integer
            days_to_expiry = row.get('DaysToExpiry')
            if pd.notna(days_to_expiry) and isinstance(days_to_expiry, (int, float)):
                days_to_expiry = int(days_to_expiry)
            else:
                days_to_expiry = None
            
            # Build strict record with ALL columns
            strict_record: Dict[str, Any] = {
                # SYSTEM/IDENTITY
                'Name': filename,
                'Content Type': 'WRT Global B2B Contract Processing',
                'ContractId': contract_id,
                'ContractTitle': contract_title,
                'ContractDuration': contract_duration,
                
                # PARTIES
                'Party1 Name': str(row.get('Party1 Name', '') or ''),
                'Party1 Address': str(row.get('Party1 Address', '') or ''),
                'Party2 Name': str(row.get('Party2 Name', '') or 'Spectralink Corporation'),
                'Party2 Address': spectralink_address,
                
                # JURISDICTION
                'Jurisdiction': str(row.get('Jurisdiction', '') or ''),
                
                # DATES (YYYY-MM-DD only)
                'EffectiveDate': format_date(effective_date),
                'RenewalDate': format_date(renewal_date),
                'ExpirationDate': format_date(expiration_date),
                'ExecutionDate': format_date(execution_date),
                
                # RENEWAL/TERMINATION TERMS
                'Autorenew': str(row.get('Autorenew', '') or ''),
                'Notification Period in days': notification_period,
                'NotificationPeriodDays': notification_period,  # Duplicate for compatibility
                'Notification_Buffer_Days': notification_buffer,
                'TerminationNotice': str(row.get('TerminationNotice', '') or ''),
                'RenewalNotice': str(row.get('RenewalNotice', '') or ''),
                'PerpetualAgreement': str(row.get('PerpetualAgreement', '') or ''),
                'PerpetualTerminationNoticeDays': perpetual_notice_days,
                
                # FLAGS & COMPLIANCE
                'Audit Flag': str(row.get('Audit Flag', '') or ''),
                'AuditReason': str(row.get('AuditReason', '') or ''),
                'Lifecycle Stage': str(row.get('Lifecycle Stage', '') or ''),
                'Risk Level': str(row.get('Risk Level', '') or ''),
                'Archive Flag': str(row.get('Archive Flag', '') or ''),
                'ArchiveReason': str(row.get('ArchiveReason', '') or ''),
                'GDPR_Applicable': str(row.get('GDPR_Applicable', '') or ''),
                'Department': str(row.get('Department', '') or ''),
                
                # OWNERSHIP & VALUE
                'Contract_Owner': str(row.get('Contract_Owner', '') or ''),
                'Contract_CoOwner': str(row.get('Contract_CoOwner', '') or ''),
                'ContractValue': contract_value_numeric,
                'ContractCurrency': contract_currency,
                'ContractSummary': str(row.get('ContractSummary', '') or '')[:500] if row.get('ContractSummary') else '',
                'InitialTerm': str(row.get('InitialTerm', '') or ''),
                'RenewalTerm': str(row.get('RenewalTerm', '') or ''),
                
                # HIERARCHY & FAMILY
                'CounterpartyRegion': region,
                'ContractFamily': str(row.get('ContractFamily', '') or ''),
                'ParentContractId': parent_id,
                'ContractHierarchy': hierarchy,
                'DaysToExpiry': days_to_expiry,
                'TermsInherited': str(row.get('TermsInherited', '') or ''),
                'SOWStartDate': format_date(sow_start),
                'SOWEndDate': format_date(sow_end),
                'ParentAgreementDate': parent_agreement_date,
            }
            
            records.append(strict_record)
        
        # Create DataFrame with exact column order
        strict_df = pd.DataFrame(records)
        
        # Ensure ALL columns exist and are in correct order
        for col in self.STRICT_SHAREPOINT_SCHEMA:
            if col not in strict_df.columns:
                strict_df[col] = None
        
        # Reorder to match schema exactly
        strict_df = strict_df[self.STRICT_SHAREPOINT_SCHEMA]
        
        logger.info(f"✅ Strict schema applied: {len(strict_df)} rows, {len(strict_df.columns)} columns")
        return strict_df
        return strict_df
    
    def export_to_excel_strict(self, filename: Optional[str] = None) -> Optional[Path]:
        """Export to Excel file with STRICT SharePoint schema (fewer columns)"""
        df = self.export_to_dataframe()
        
        if df.empty:
            logger.warning("⚠️  No data to export")
            return None
        
        # Apply strict schema transformation
        strict_df = self._apply_strict_schema(df)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"contracts_sharepoint_strict_{timestamp}.xlsx"
        
        output_path = self.output_dir / filename
        
        # Export with Excel formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            strict_df.to_excel(cast(Any, writer), sheet_name='Contracts', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Contracts']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"✅ Exported (strict schema) to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Export contracts to SharePoint format')
    parser.add_argument('--chroma-path', default='./chroma_db', help='ChromaDB path')
    parser.add_argument('--collection', default='contracts', help='ChromaDB collection name')
    parser.add_argument('--output-dir', default='./sharepoint_exports', help='Output directory')
    parser.add_argument('--preview', type=int, default=0, help='Preview n rows (0=export)')
    parser.add_argument('--validate', action='store_true', help='Validate against SharePoint schema')
    parser.add_argument('--strict', action='store_true', help='Use strict 20-column SharePoint schema')
    parser.add_argument('--output', '--filename', dest='filename', help='Output filename')
    
    args = parser.parse_args()
    
    exporter = SharePointExporter(args.chroma_path, args.output_dir, args.collection)
    
    if args.validate:
        result = exporter.validate_schema()
        print(f"\n{'='*60}")
        print("SCHEMA VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Match: {result['match_percentage']:.1f}%")
        print(f"\nMatching columns ({len(result['matching'])}):")
        for col in sorted(result['matching']):
            print(f"  ✅ {col}")
        
        if result.get('missing_from_export'):
            print(f"\nMissing from export ({len(result['missing_from_export'])}):")
            for col in sorted(result['missing_from_export']):
                print(f"  ❌ {col}")
    
    elif args.preview > 0:
        df = exporter.export_to_dataframe()
        
        # Apply strict schema if requested
        if args.strict:
            df = exporter._apply_strict_schema(df)
            preview_cols = exporter.STRICT_SHAREPOINT_SCHEMA
        else:
            # Full schema preview columns
            preview_cols = [
                'ContractId', 'ContractHierarchy', 'ParentContractId', 'ContractIdNotes',
                'Name', 'Party1 Name', 'Party2 Name', 
                'EffectiveDate', 'ExpirationDate', 'ExecutionDate', 'ContractDuration',
                'Lifecycle Stage', 'Risk Level', 'Department',
                'Contract_Owner', 'Contract_CoOwner',
                'GDPR_Applicable', 'GDPRReason',
                'Audit Flag', 'AuditReason',
                'Archive Flag', 'ArchiveReason',
                'Autorenew', 'Notification Period in days', 'Notification_Buffer_Days',
                'InitialTerm', 'RenewalTerm',
                'TerminationNotice', 'RenewalNotice',
                'PerpetualAgreement', 'PerpetualNotice', 'PerpetualNoticeDays',
                'ReviewFlag', 'FolderStatusNotes', 'CounterpartyRegion',
                'Jurisdiction', 'ContractValue',
                'ContractFamily', 
                'DaysToExpiry', 'TermsInherited',
                'SOWStartDate', 'SOWEndDate', 'ParentAgreementDate',
            ]
        
        df = df.head(args.preview)
        schema_type = "STRICT" if args.strict else "FULL"
        print(f"\n{'='*60}")
        print(f"PREVIEW ({schema_type}): First {len(df)} rows ({len(df.columns)} columns)")
        print(f"{'='*60}")
        
        # Print each row in a readable format
        for row_num, (_, row) in enumerate(df.iterrows(), start=1):
            print(f"\n--- Contract {row_num} ---")
            for col in preview_cols:
                if col not in df.columns:
                    continue
                val = row.get(col, 'N/A')
                # Format values nicely
                if pd.notna(val):
                    if hasattr(val, 'strftime'):
                        val = val.strftime('%Y-%m-%d')
                    elif val == '' or (isinstance(val, float) and pd.isna(val)):
                        val = '(empty)'
                    elif col == 'DaysToExpiry' and isinstance(val, (int, float)):
                        val = int(val)
                    elif col in ['Notification Period in days', 'Notification_Buffer_Days', 'PerpetualNoticeDays']:
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            val = int(val)
                    print(f"  {col}: {val}")
                else:
                    print(f"  {col}: (not set)")
    
    else:
        # Export mode
        if args.strict:
            output_path = exporter.export_to_excel_strict(args.filename)
        else:
            output_path = exporter.export_to_excel(args.filename)
            
        if output_path:
            print(f"\n{'='*60}")
            print(f"✅ EXPORT COMPLETE {'(STRICT SCHEMA)' if args.strict else ''}")
            print(f"{'='*60}")
            print(f"Output: {output_path}")
            print(f"\nNext steps:")
            print(f"  1. Open in Excel to review")
            print(f"  2. Import to SharePoint list")
            print(f"  3. Set up Power Automate flows")


if __name__ == "__main__":
    main()
