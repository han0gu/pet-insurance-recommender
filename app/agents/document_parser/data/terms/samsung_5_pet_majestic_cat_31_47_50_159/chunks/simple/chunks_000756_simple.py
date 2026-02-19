from langchain_core.documents import Document

chunk = Document(
    page_content=('. 대한민국 이외 지역에서 발생한 사고 및 손해 12. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행 '
 '위로 인한 손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 그로 인하여 가중 된 손해 13. 국가 및 지방자치단체의 명령 또는 '
 '법률에 의한 살처분 또는 이와 유사한 사태'),
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
 'indexing': {'chunk_id': 'chunk_000756',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
