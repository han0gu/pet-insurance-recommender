from langchain_core.documents import Document

chunk = Document(
    page_content=('. 통약 1. 실종선고를 받은 경우 : 법원에서 인정한 실종기간이 끝나는 때에 사망한 것으 관 로 봅니다. 2. 관공서에서 수해, 화재나 '
 '그 밖의 재난을 조사하고 사망한 것으로 통보하는 경 |'),
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
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
