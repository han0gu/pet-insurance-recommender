from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중 에 보험증권에 기재된 반려묘가 국내에서 수의사에게 '
 '이물 섭취 치료를 목적으로 이 물제거(내시경) 또는 이물제거(구토유도약물)에 해당하는 치료를 받은경우 연간2회 에 한하여 당일 피보험자가 '
 '부담한 반려묘의 치료에 사용된 비용(각종 할인 및감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하「의료비」 라 합니다)을 제3 '
 '항에 따라보험가입금액을 한도로 보험수익자에게 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000700',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
