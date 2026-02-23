from langchain_core.documents import Document

chunk = Document(
    page_content=('| ∙ 보험가입금액 제한 피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다. ∙ 일부보장 제외 일반적인 경우보다 '
 '위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 특 정 질병 또는 특정 신체 부위를 보장에서 제외하는 방법을 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
