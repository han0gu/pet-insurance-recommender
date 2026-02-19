from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 아래의 의료비 및 비용 또는 손해는 보상하지 않습니다. 1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 '
 '접종비용 및 정기검'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000552',
              'chunk_char_len': 86,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
