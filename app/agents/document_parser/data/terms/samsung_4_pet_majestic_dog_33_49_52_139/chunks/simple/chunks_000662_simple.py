from langchain_core.documents import Document

chunk = Document(
    page_content='<용어풀이>\n배꼽허니아 | 복부 내장의 탈장 등으로 인해 배꼽 주변이 부풀어 오르는 증상',
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other', 'other']},
 'indexing': {'chunk_id': 'chunk_000662',
              'chunk_char_len': 49,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
