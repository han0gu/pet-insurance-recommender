from langchain_core.documents import Document

chunk = Document(
    page_content=('니다) 중에 상해의 직접적인 결과로써 사망한 경우(질병으로 인한 사망은 제외합니다) 5\n'
 '년간 매월 보험증권에 기재된 이 특별약관의 보험가입금액을 보험금 지급사유 발생일(단,\n'
 '해당월에 보험금 지급사유 발생일이 없는 경우에는 해당월의 마지막 날로 합니다)에 반려\n'
 '동물 양육자금Ⅰ으로 보험수익자에게 지급합니다.# 제 2조 (보험금 지급에 관한 세부규정)① 제1조(보험금의 지급사유)의‘사망’에는 '
 '보험기간에 다음 어느 하나의 사유가 발생'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000395',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
