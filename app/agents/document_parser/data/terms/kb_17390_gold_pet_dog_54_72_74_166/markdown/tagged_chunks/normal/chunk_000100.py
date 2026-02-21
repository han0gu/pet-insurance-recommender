from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 보험료 할증 일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보 험 가입 후 기간이 경과함에 따라 위험의 '
 '크기 및 정도가 점차 증가하는 위험 또는 기간의 경과에 상관없이 일정한 상태를 유지하는 위험에 적용하는 방법으 로 위험 정도에 따라 '
 '특별보험료를 추가로 부가하는 방법을 말합니다. | ∙ 보험가입금액 제한 피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 '
 '말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
