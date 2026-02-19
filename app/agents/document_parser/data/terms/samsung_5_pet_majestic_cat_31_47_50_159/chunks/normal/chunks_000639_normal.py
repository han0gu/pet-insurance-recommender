from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험계약이 연장된 경우 연장 된 날 기준으로 매년 현재의 예정기초율(적용이율, 적용위험률, 부가보험요율) 적용 및 반려동물의 '
 '연령 증가 등의 사유로 보험요율이 변동될 수 있으며 이 때의 보험료 는 「보험료 및 해약환급금 산출방법서」 에 따라 산출합니다. 또한, '
 '보험계약의 연장은 기본계약의 보험기간 내에서만 가능합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000639',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
