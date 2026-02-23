from langchain_core.documents import Document

chunk = Document(
    page_content=('| 말합니다. 예시) 보험계약일이 8월 15일인 경우 보험년도 기준 매1년은 해당년도 8월 15 일부터 다음 해 8월 14일까지입니다. '
 '| 말합니다. 예시) 보험계약일이 8월 15일인 경우 보험년도 기준 매1년은 해당년도 8월 15 일부터 다음 해 8월 14일까지입니다. '
 '| 말합니다. 예시) 보험계약일이 8월 15일인 경우 보험년도 기준 매1년은 해당년도 8월 15 일부터 다음 해 8월 14일까지입니다. '
 '| 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
