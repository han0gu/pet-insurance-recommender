from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부담보 기간」이라 합니다)은 특 정신체부위 또는 특정질병의 상태에 따라 '
 '「1개월부터 5년」또는「보험계약의 보험기 간 전체」로 하며, 그 판단기준은 회사에서 정한 계약사정기준을 따릅니다. 다만, 개 개인의 '
 '질병의 상태 등에 대한 의사의 소견에 따라서 다르게 적용할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 136},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000852',
              'chunk_char_len': 184,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
