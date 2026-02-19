from langchain_core.documents import Document

chunk = Document(
    page_content=('. 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다. 파) 외상후 스트레스장애, 우울증(반응성) 등의 질환, '
 '정신분열증(조현병), 편집 증, 조울증(양극성장애), 불안장애, 전환장애, 공포장애, 강박장애 등 각종 신경증 및 각종 인격장애는 보상의 '
 '대상이 되지 않는다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000978',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
