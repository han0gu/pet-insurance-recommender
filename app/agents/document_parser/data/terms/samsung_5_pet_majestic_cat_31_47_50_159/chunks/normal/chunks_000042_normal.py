from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험년도]\n'
 '당해연도 보험계약 해당일부터 다음년도 보험계약 해당일 전일까지로 매1년 단위의 연도임. 예를 들어, 보험계약일이 2022년 4월 1일인 '
 '경우 보험년도는 4월 1일부터 2023년도 3월 31일까지 1년 을 말함\n'
 '<예시안내>\n'
 '[중도인출금의 한도 예시] 중도인출 시점에서 계산된 기본계약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 금액이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 203,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
