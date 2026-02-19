from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조 (보험금 지급에 관한 세부규정)\n'
 '① 피보험자가 「국민건강보험법」 또는 「의료급여법」 을 적용받지 못하는 사고로 인하 여 창상봉합술을 받은 경우, 진단서 및 '
 '진료비세부내역서 등을 통해 이 특별약관에서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000509',
              'chunk_char_len': 114,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
