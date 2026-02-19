from langchain_core.documents import Document

chunk = Document(
    page_content=('. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환, '
 '배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 손톱깎기 등의 처치비용'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000759',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
