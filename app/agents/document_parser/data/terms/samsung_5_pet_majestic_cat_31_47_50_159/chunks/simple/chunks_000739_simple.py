from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하「의료비」 라 합니다 )을 제4항에 따라 보험가입금액을 한도로 보험수익자에게 보상하여 드립니다. ② 반려묘가 제1항의 사고로 '
 '치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 180일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 '
 '365일이내 의 치료인 경우에 한합니다. ③ 제1항의 손해에 대한 보장개시일(책임개시일)은 이 특별약관의 보험계약일(이하 「보 '
 '험계약일」이라 합니다)부터 그 날을 포함하여 30일이 지난날의 다음날로 합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000739',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
