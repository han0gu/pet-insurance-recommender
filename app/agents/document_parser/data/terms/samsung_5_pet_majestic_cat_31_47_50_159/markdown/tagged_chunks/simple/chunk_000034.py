from langchain_core.documents import Document

chunk = Document(
    page_content=('익률과 외부지표금리를 가중평균하여 산출된 공시기준이율에서 향후 예상수익 등을\n'
 '고려한 조정률을 가감하여 결정합니다.<용어풀이># [운용자산이익률]운용자산수익률에서 투자지출률을 차감하여 산출합니다.\n'
 '[외부지표금리]\n'
 '국고채 수익률, 회사채 수익률, 통화안정증권 수익률, 양도성예금증서 유통수익률을 기준으로 산출\n'
 '합니다.③ 회사는 제1항 및 제2항에서 정한 공시이율 및 산출방법 등을 매월 회사의 인터넷 홈'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000034',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
