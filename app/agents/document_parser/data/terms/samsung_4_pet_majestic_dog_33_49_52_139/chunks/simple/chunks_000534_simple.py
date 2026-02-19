from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중 에 제3항에서 정한 보장개시일(책임개시일) 이후에 '
 '보험증권에 기재된 반려견에게 상 해 또는 질병(이하 「사고」 라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국내 에서 수의사에게 '
 '치료를 받은 때에는 수술을 받지 않은 날에 한하여 1일당 피보험자 가 부담한 반려견의 치료에 사용된 비용(각종 할인 및 감면, '
 '사후환급금액 등을 제외 한 실수납액을 의미합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000534',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
