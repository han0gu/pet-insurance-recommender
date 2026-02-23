from langchain_core.documents import Document

chunk = Document(
    page_content=('- 말합니다.\n'
 '# 제3조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중\n'
 '- 에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려묘에게 상\n'
 '- 해 또는 질병(이하 「사고」 라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국내\n'
 '- 에서 수의사에게 치료를 받은 때에는 1일당 피보험자가 부담한 반려묘의 치료에 사용\n'
 '- 된 비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000453',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
