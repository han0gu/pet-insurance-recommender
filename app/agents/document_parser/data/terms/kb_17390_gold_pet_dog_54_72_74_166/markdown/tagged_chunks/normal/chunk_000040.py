from langchain_core.documents import Document

chunk = Document(
    page_content=('- 급금의 지급 시점까지 인출금액에 적립되었을 이자만큼 만기환급금이 감소합니다. ㆍ\n'
 '- 57 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 57규정|  |\n'
 '| --- |'),
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
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
