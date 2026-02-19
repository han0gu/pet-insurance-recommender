from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 이 추가특별약관의 보험기간(이하 「보험기간」 이라 합니다 ) 중에 제3항에서 정한 보장개시일(책임개시일) '
 '이후에 보험증권에 기재된 반려묘에게 상해 또는 질병(이하 「사고」 라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국 내에서 '
 '수의사에게 수술을 받은 경우 연간 2회에 한하여 피보험자가 부담한 수술 당 일 반려묘의 치료에 사용된 비용(각종 할인 및 감면, '
 '사후환급금액 등을 제외한 실수 납액을 의미합니다. 이하 「의료비」 라 합니다)을 제4항에 따라 보험가입금액을 한도 로 보험수익자에게 '
 '4-1'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000646',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
