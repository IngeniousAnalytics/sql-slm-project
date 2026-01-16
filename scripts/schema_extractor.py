"""
Schema Extractor - Automatically extracts database schema information

This script:
1. Reads database credentials from .env file
2. Auto-detects database type and version
3. Extracts complete schema (tables, columns, keys, relationships)
4. Saves schema information to JSON file

Usage:
    python schema_extractor.py
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
    # Load .env at import time and allow overriding existing environment variables
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path, override=True)
except ImportError:
    def load_dotenv(*args, **kwargs):
        # Fallback no-op loader if python-dotenv is not installed
        return False
    _DOTENV_AVAILABLE = False

# Import database connectors
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

try:
    import pymysql
except ImportError:
    pymysql = None

try:
    import pyodbc
except ImportError:
    pyodbc = None

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None

import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Warn if python-dotenv is unavailable
if not globals().get("_DOTENV_AVAILABLE", True):
    logger.warning("python-dotenv not installed; .env will not be loaded. Install with: pip install python-dotenv")


class DatabaseConnector:
    """Handles connections to different database types"""
    
    def __init__(self):
        # Ensure latest .env values are loaded, overriding any existing env vars
        try:
            load_dotenv(find_dotenv(), override=True)  # type: ignore
        except Exception:
            # If python-dotenv isn't available or find_dotenv isn't defined, fall back silently
            load_dotenv()
        self.db_type = os.getenv('DB_TYPE', 'postgresql').lower()
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection based on DB_TYPE"""
        logger.info(f"Connecting to {self.db_type} database...")
        
        try:
            if self.db_type == 'postgresql':
                self._connect_postgresql()
            elif self.db_type == 'mysql':
                self._connect_mysql()
            elif self.db_type == 'sqlserver':
                self._connect_sqlserver()
            elif self.db_type == 'oracle':
                self._connect_oracle()
            elif self.db_type == 'sqlite':
                self._connect_sqlite()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            logger.info("✓ Successfully connected to database")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to connect: {str(e)}")
            return False
    
    def _connect_postgresql(self):
        """Connect to PostgreSQL database"""
        if not psycopg2:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")
        
        self.connection = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=int(os.getenv('POSTGRES_PORT', 5432)),
            database=os.getenv('POSTGRES_DATABASE'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
    
    def _connect_mysql(self):
        """Connect to MySQL database"""
        if not pymysql:
            raise ImportError("PyMySQL not installed. Run: pip install PyMySQL")
        
        self.connection = pymysql.connect(
            host=os.getenv('MYSQL_HOST'),
            port=int(os.getenv('MYSQL_PORT', 3306)),
            database=os.getenv('MYSQL_DATABASE'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD')
        )
        self.cursor = self.connection.cursor(pymysql.cursors.DictCursor)
    
    def _connect_sqlserver(self):
        """Connect to SQL Server database"""
        if not pyodbc:
            raise ImportError("pyodbc not installed. Run: pip install pyodbc")
        
        driver = os.getenv('SQLSERVER_DRIVER', 'ODBC Driver 17 for SQL Server')
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={os.getenv('SQLSERVER_HOST')};"
            f"PORT={os.getenv('SQLSERVER_PORT', 1433)};"
            f"DATABASE={os.getenv('SQLSERVER_DATABASE')};"
            f"UID={os.getenv('SQLSERVER_USER')};"
            f"PWD={os.getenv('SQLSERVER_PASSWORD')}"
        )
        self.connection = pyodbc.connect(conn_str)
        self.cursor = self.connection.cursor()
    
    def _connect_oracle(self):
        """Connect to Oracle database"""
        if not cx_Oracle:
            raise ImportError("cx_Oracle not installed. Run: pip install cx_Oracle")
        
        dsn = cx_Oracle.makedsn(
            os.getenv('ORACLE_HOST'),
            int(os.getenv('ORACLE_PORT', 1521)),
            service_name=os.getenv('ORACLE_SERVICE_NAME')
        )
        self.connection = cx_Oracle.connect(
            user=os.getenv('ORACLE_USER'),
            password=os.getenv('ORACLE_PASSWORD'),
            dsn=dsn
        )
        self.cursor = self.connection.cursor()
    
    def _connect_sqlite(self):
        """Connect to SQLite database"""
        db_path = os.getenv('SQLITE_PATH')
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database not found: {db_path}")
        
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
    
    def execute(self, query: str, params: tuple = None):
        """Execute SQL query and return results as a list of dicts with normalized keys.

        This method normalizes column names to lowercase, handles driver-dependent
        row types (tuples, dict-like, sqlite3.Row), and filters out accidental
        header-like rows where values equal column names (e.g., {'name': 'name'}).
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)

            # Return results as list of dictionaries
            if self.cursor.description:
                columns = [desc[0] for desc in self.cursor.description]
                raw_rows = self.cursor.fetchall()

                results: List[Dict[str, Any]] = []
                for row in raw_rows:
                    # If row is a mapping-like (e.g., dict, RealDictRow), try to map by column names
                    if isinstance(row, dict):
                        normalized: Dict[str, Any] = {}
                        for col in columns:
                            # Prefer the exact column name, then lower/upper variants
                            if col in row:
                                normalized[col.lower()] = row[col]
                            elif col.lower() in row:
                                normalized[col.lower()] = row[col.lower()]
                            elif col.upper() in row:
                                normalized[col.lower()] = row[col.upper()]
                            else:
                                normalized[col.lower()] = None
                        results.append(normalized)
                    else:
                        # Sequence-like (tuple, list, sqlite3.Row)
                        results.append({columns[i].lower(): row[i] for i in range(len(columns))})

                # Detect and filter out header-like rows (where every value equals its column name)
                def is_header_like(r: Dict[str, Any]) -> bool:
                    if not r:
                        return False
                    for k, v in r.items():
                        if not isinstance(v, str):
                            return False
                        if v.strip().lower() != k.lower():
                            return False
                    return True

                cleaned = [r for r in results if not is_header_like(r)]
                if cleaned and len(cleaned) != len(results):
                    logger.info("Filtered out header-like row(s) from query results")
                    return cleaned
                return results

            return []

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")


class SchemaExtractor:
    """Extracts complete database schema information"""
    
    def __init__(self, connector: DatabaseConnector):
        self.connector = connector
        self.db_type = connector.db_type
        # By default, do NOT include table rows or counts. Set INCLUDE_TABLE_ROWS=1/true to enable.
        self.include_rows = os.getenv('INCLUDE_TABLE_ROWS', 'false').lower() in ('1', 'true', 'yes')
        logger.info(f"Include table rows: {self.include_rows}")

        self.schema_data = {
            "extraction_date": datetime.now().isoformat(),
            "database_type": self.db_type,
            "database_version": "",
            "database_name": "",
            "tables": []
        }
    
    def extract_complete_schema(self) -> Dict:
        """Main method to extract all schema information"""
        logger.info("Starting schema extraction...")
        
        # Step 1: Get database version and info
        self._get_database_info()
        
        # Step 2: Get all tables
        tables = self._get_tables()
        logger.info(f"Found {len(tables)} tables")
        
        # Step 3: For each table, get detailed information
        for idx, table_info in enumerate(tables, 1):
            logger.info(f"Processing table {idx}/{len(tables)}: {table_info['name']}")
            
            # Base metadata for the table (no actual row data by default)
            table_data = {
                "name": table_info['name'],
                "schema": table_info.get('schema', 'public'),
                "description": table_info.get('description', ''),
                "columns": self._get_columns(table_info['name']),
                "primary_keys": self._get_primary_keys(table_info['name']),
                "foreign_keys": self._get_foreign_keys(table_info['name']),
                "indexes": self._get_indexes(table_info['name'])
            }

            # Optionally include row count and a small sample of rows (opt-in via env var)
            if self.include_rows:
                table_data['row_count'] = self._get_row_count(table_info['name'])
                table_data['sample_data'] = self._get_sample_data(table_info['name'])
            
            self.schema_data['tables'].append(table_data)
        
        logger.info("✓ Schema extraction completed")
        return self.schema_data
    
    def _get_database_info(self):
        """Get database version and name"""
        logger.info("Extracting database information...")
        
        if self.db_type == 'postgresql':
            result = self.connector.execute("SELECT version();")
            if result:
                self.schema_data['database_version'] = result[0].get('version', '')
            
            # Prefer the database name from the environment (e.g., .env) if provided
            # (support common names used in Docker images: POSTGRES_DATABASE and POSTGRES_DB).
            env_db = os.getenv('POSTGRES_DATABASE') or os.getenv('POSTGRES_DB')
            if env_db:
                self.schema_data['database_name'] = env_db
            else:
                result = self.connector.execute("SELECT current_database();")
                if result:
                    self.schema_data['database_name'] = result[0].get('current_database', '')
        
        elif self.db_type == 'mysql':
            result = self.connector.execute("SELECT VERSION();")
            if result:
                self.schema_data['database_version'] = list(result[0].values())[0]
            
            result = self.connector.execute("SELECT DATABASE();")
            if result:
                self.schema_data['database_name'] = list(result[0].values())[0]
        
        elif self.db_type == 'sqlserver':
            result = self.connector.execute("SELECT @@VERSION;")
            if result:
                self.schema_data['database_version'] = list(result[0].values())[0]
            
            result = self.connector.execute("SELECT DB_NAME();")
            if result:
                self.schema_data['database_name'] = list(result[0].values())[0]
        
        elif self.db_type == 'oracle':
            result = self.connector.execute("SELECT * FROM v$version WHERE banner LIKE 'Oracle%'")
            if result:
                self.schema_data['database_version'] = list(result[0].values())[0]
            
            result = self.connector.execute("SELECT SYS_CONTEXT('USERENV', 'DB_NAME') FROM dual")
            if result:
                self.schema_data['database_name'] = list(result[0].values())[0]
        
        elif self.db_type == 'sqlite':
            result = self.connector.execute("SELECT sqlite_version();")
            if result:
                self.schema_data['database_version'] = f"SQLite {list(result[0].values())[0]}"
            
            self.schema_data['database_name'] = os.path.basename(os.getenv('SQLITE_PATH', 'unknown'))
    
    def _get_tables(self) -> List[Dict]:
        """Get all tables in the database"""
        tables = []
        
        if self.db_type == 'postgresql':
            query = """
                SELECT 
                    t.table_schema as schema,
                    t.table_name as name,
                    obj_description((quote_ident(t.table_schema)||'.'||quote_ident(t.table_name))::regclass, 'pg_class') as description
                FROM information_schema.tables t
                WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
                AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_schema, t.table_name;
            """
            results = self.connector.execute(query)
            tables = [{'name': r['name'], 'schema': r['schema'], 'description': r.get('description') or ''} 
                     for r in results]
        
        elif self.db_type == 'mysql':
            query = f"""
                SELECT 
                    TABLE_NAME as name,
                    TABLE_SCHEMA as `schema`,
                    TABLE_COMMENT as description
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = '{os.getenv('MYSQL_DATABASE')}'
                AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME;
            """
            results = self.connector.execute(query)
            tables = [{'name': r['name'], 'schema': r['schema'], 'description': r.get('description') or ''} 
                     for r in results]
        
        elif self.db_type == 'sqlserver':
            query = """
                SELECT 
                    SCHEMA_NAME(t.schema_id) as [schema],
                    t.name,
                    ep.value as description
                FROM sys.tables t
                LEFT JOIN sys.extended_properties ep 
                    ON ep.major_id = t.object_id 
                    AND ep.minor_id = 0 
                    AND ep.name = 'MS_Description'
                ORDER BY t.name;
            """
            results = self.connector.execute(query)
            tables = [{'name': r['name'], 'schema': r['schema'], 'description': r.get('description') or ''} 
                     for r in results]
        
        elif self.db_type == 'oracle':
            query = """
                SELECT 
                    table_name as name,
                    owner as schema,
                    '' as description
                FROM all_tables
                WHERE owner = USER
                ORDER BY table_name
            """
            results = self.connector.execute(query)
            tables = [{'name': r['name'], 'schema': r['schema'], 'description': ''} 
                     for r in results]
        
        elif self.db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            results = self.connector.execute(query)
            tables = [{'name': r['name'], 'schema': 'main', 'description': ''} 
                     for r in results]
        
        return tables
    
    def _get_columns(self, table_name: str) -> List[Dict]:
        """Get all columns for a table"""
        columns = []
        
        if self.db_type == 'postgresql':
            query = f"""
                SELECT 
                    c.column_name as name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default as default_value,
                    c.character_maximum_length,
                    col_description((quote_ident(c.table_schema)||'.'||quote_ident(c.table_name))::regclass, c.ordinal_position) as description
                FROM information_schema.columns c
                WHERE c.table_name = '{table_name}'
                ORDER BY c.ordinal_position;
            """
            results = self.connector.execute(query)
            
            for r in results:
                columns.append({
                    'name': r['name'],
                    'data_type': r['data_type'],
                    'nullable': r['is_nullable'] == 'YES',
                    'default': r.get('default_value'),
                    'max_length': r.get('character_maximum_length'),
                    'description': r.get('description') or ''
                })
        
        elif self.db_type == 'mysql':
            query = f"""
                SELECT 
                    COLUMN_NAME as name,
                    DATA_TYPE as data_type,
                    IS_NULLABLE as is_nullable,
                    COLUMN_DEFAULT as default_value,
                    CHARACTER_MAXIMUM_LENGTH as max_length,
                    COLUMN_COMMENT as description
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                AND TABLE_SCHEMA = '{os.getenv('MYSQL_DATABASE')}'
                ORDER BY ORDINAL_POSITION;
            """
            results = self.connector.execute(query)
            
            for r in results:
                columns.append({
                    'name': r['name'],
                    'data_type': r['data_type'],
                    'nullable': r['is_nullable'] == 'YES',
                    'default': r.get('default_value'),
                    'max_length': r.get('max_length'),
                    'description': r.get('description') or ''
                })
        
        elif self.db_type == 'sqlserver':
            query = f"""
                SELECT 
                    c.name,
                    t.name as data_type,
                    c.is_nullable,
                    dc.definition as default_value,
                    c.max_length,
                    ep.value as description
                FROM sys.columns c
                INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
                LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id
                LEFT JOIN sys.extended_properties ep ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
                WHERE c.object_id = OBJECT_ID('{table_name}')
                ORDER BY c.column_id;
            """
            results = self.connector.execute(query)
            
            for r in results:
                columns.append({
                    'name': r['name'],
                    'data_type': r['data_type'],
                    'nullable': bool(r['is_nullable']),
                    'default': r.get('default_value'),
                    'max_length': r.get('max_length'),
                    'description': r.get('description') or ''
                })
        
        elif self.db_type == 'sqlite':
            query = f"PRAGMA table_info({table_name});"
            results = self.connector.execute(query)
            
            for r in results:
                columns.append({
                    'name': r['name'],
                    'data_type': r['type'],
                    'nullable': not bool(r['notnull']),
                    'default': r.get('dflt_value'),
                    'max_length': None,
                    'description': ''
                })
        
        return columns
    
    def _get_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns"""
        pks = []
        
        if self.db_type == 'postgresql':
            query = f"""
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = '{table_name}'
                AND tc.constraint_type = 'PRIMARY KEY'
                ORDER BY kcu.ordinal_position;
            """
            results = self.connector.execute(query)
            pks = [r['column_name'] for r in results]
        
        elif self.db_type == 'mysql':
            query = f"""
                SELECT COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_NAME = '{table_name}'
                AND TABLE_SCHEMA = '{os.getenv('MYSQL_DATABASE')}'
                AND CONSTRAINT_NAME = 'PRIMARY'
                ORDER BY ORDINAL_POSITION;
            """
            results = self.connector.execute(query)
            pks = [r['COLUMN_NAME'] for r in results]
        
        elif self.db_type == 'sqlite':
            query = f"PRAGMA table_info({table_name});"
            results = self.connector.execute(query)
            pks = [r['name'] for r in results if r['pk'] > 0]
        
        return pks
    
    def _get_foreign_keys(self, table_name: str) -> List[Dict]:
        """Get foreign key relationships"""
        fks = []
        
        if self.db_type == 'postgresql':
            query = f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.table_name = '{table_name}'
                AND tc.constraint_type = 'FOREIGN KEY';
            """
            results = self.connector.execute(query)
            
            for r in results:
                fks.append({
                    'column': r['column_name'],
                    'references_table': r['foreign_table'],
                    'references_column': r['foreign_column']
                })
        
        elif self.db_type == 'mysql':
            query = f"""
                SELECT
                    COLUMN_NAME as column_name,
                    REFERENCED_TABLE_NAME as foreign_table,
                    REFERENCED_COLUMN_NAME as foreign_column
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_NAME = '{table_name}'
                AND TABLE_SCHEMA = '{os.getenv('MYSQL_DATABASE')}'
                AND REFERENCED_TABLE_NAME IS NOT NULL;
            """
            results = self.connector.execute(query)
            
            for r in results:
                fks.append({
                    'column': r['column_name'],
                    'references_table': r['foreign_table'],
                    'references_column': r['foreign_column']
                })
        
        elif self.db_type == 'sqlite':
            query = f"PRAGMA foreign_key_list({table_name});"
            results = self.connector.execute(query)
            
            for r in results:
                fks.append({
                    'column': r['from'],
                    'references_table': r['table'],
                    'references_column': r['to']
                })
        
        return fks
    
    def _get_indexes(self, table_name: str) -> List[Dict]:
        """Get table indexes"""
        indexes = []
        
        if self.db_type == 'postgresql':
            query = f"""
                SELECT
                    i.relname as index_name,
                    array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) as columns,
                    ix.indisunique as is_unique
                FROM pg_class t
                JOIN pg_index ix ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                WHERE t.relname = '{table_name}'
                GROUP BY i.relname, ix.indisunique;
            """
            results = self.connector.execute(query)
            
            for r in results:
                indexes.append({
                    'name': r['index_name'],
                    'columns': r['columns'],
                    'unique': r['is_unique']
                })
        
        elif self.db_type == 'sqlite':
            query = f"PRAGMA index_list({table_name});"
            results = self.connector.execute(query)
            
            for idx in results:
                idx_info_query = f"PRAGMA index_info({idx['name']});"
                idx_cols = self.connector.execute(idx_info_query)
                
                indexes.append({
                    'name': idx['name'],
                    'columns': [col['name'] for col in idx_cols],
                    'unique': bool(idx['unique'])
                })
        
        return indexes
    
    def _get_row_count(self, table_name: str) -> int:
        """Get approximate row count"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name};"
            result = self.connector.execute(query)
            return result[0]['count'] if result else 0
        except:
            return 0
    
    def _get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample rows from table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            results = self.connector.execute(query)
            
            # Convert to JSON-serializable format
            serialized = []
            for row in results:
                serialized_row = {}
                for key, value in row.items():
                    # Convert non-serializable types to strings
                    if isinstance(value, (datetime, bytes)):
                        serialized_row[key] = str(value)
                    else:
                        serialized_row[key] = value
                serialized.append(serialized_row)
            
            return serialized
        except Exception as e:
            logger.warning(f"Could not get sample data for {table_name}: {str(e)}")
            return []
    
    def save_schema(self, output_path: str = None):
        """Save schema to JSON file"""
        if output_path is None:
            output_path = os.getenv('SCHEMA_OUTPUT_PATH', './data/schemas/schema.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.schema_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Schema saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("SCHEMA EXTRACTION SUMMARY")
        print("="*60)
        print(f"Database Type: {self.schema_data['database_type']}")
        print(f"Database Version: {self.schema_data['database_version']}")
        print(f"Database Name: {self.schema_data['database_name']}")
        print(f"Total Tables: {len(self.schema_data['tables'])}")
        print(f"Output File: {output_path}")
        print("="*60)


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("SQL SLM - DATABASE SCHEMA EXTRACTOR")
    print("="*60 + "\n")
    
    # Step 1: Connect to database
    connector = DatabaseConnector()
    if not connector.connect():
        logger.error("Failed to connect to database. Check your .env configuration.")
        sys.exit(1)
    
    # Step 2: Extract schema
    extractor = SchemaExtractor(connector)
    schema = extractor.extract_complete_schema()
    
    # Step 3: Save schema
    extractor.save_schema()
    
    # Step 4: Close connection
    connector.close()
    
    print("\n✓ Schema extraction completed successfully!\n")


if __name__ == "__main__":
    main()
