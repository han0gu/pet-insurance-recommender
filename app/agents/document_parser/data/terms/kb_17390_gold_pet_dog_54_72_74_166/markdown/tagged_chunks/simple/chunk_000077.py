from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">유 의 사 항 계약자 또는 피보험자는 상해보험계약을 맺은 후 피보험자가 직업 또는 직무를 변경(자가용운전자가 '
 '영업용운전자로 직업 또는 직무 변경 포함)하거나 이륜자 동차 또는 원동기장치 자전거를 계속적으로 사용하게 된 경우에는 즉시 회사에 알려야 '
 '합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000077',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
