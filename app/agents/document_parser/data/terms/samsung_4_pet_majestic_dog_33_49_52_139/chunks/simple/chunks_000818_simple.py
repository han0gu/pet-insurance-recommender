from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 회사는 갱신전 계약의 보험기간이 만료되는 날 이전까지 중요사항 변경내역(갱신보 험료 변경 제외), 자동갱신 의사를 확인하는 내용 '
 '등을 서면(등기우편 등), 전화(음 성녹음), 전자문서(SMS 포함) 또는 이에 준하는 전자적 의사표시 등으로 2회 이상'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000818',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
