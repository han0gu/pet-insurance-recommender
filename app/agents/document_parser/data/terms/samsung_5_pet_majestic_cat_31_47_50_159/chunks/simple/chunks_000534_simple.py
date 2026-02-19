from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 「의료비」 라 합니다)을 제4항에 따라 보험가입금액을 한도로 보험수익자에게 반려묘 의료비(치과및구강질환포함)(재가입형) '
 '보험금(이하 「의료비보험금」 라 합니다)으로 보상하여 드립니다. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보 험금의 '
 '총 합계는 보험증권에 기재된 연간 총 보상한도액을 한도로 합니다. ② 반려묘가 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 '
 '경우에도 만료일부터 180일 이내의 의료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 365일이내 의 치료인 경우에 한합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 97},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000534',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
