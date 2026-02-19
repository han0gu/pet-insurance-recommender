from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 이 경우 동 물병원에서 발급한 소견서를 제출하여야 합니다. ③ 제1항의 손해에 대한 보장개시일(책임개시일)은 이 특별약관의 '
 '보험계약일(이하 「보 험계약일」 이라 합니다)부터 그 날을 포함하여 30일이 지난 날의 다음날로 합니다. 이 경우 보험계약일은 이 '
 '특별약관의 제1회 보험료를 받은 날로 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000684',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
