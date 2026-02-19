from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자 또는 피보험자는 상해보험계약을 맺은 후 피보험자가 직업 또는 직무를 변경(자가용운전자 가 영업용운전자로 직업 또는 직무 변경 '
 '포함)하거나 이륜자동차 또는 원동기장치 자전거를 계속 적으로 사용하게 된 경우에는 즉시 회사에 알려야 합니다. 그러지 않을 경우 '
 '보험사고가 발생한 경 우에도 보험금 지급이 제한될 수 있습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
