from langchain_core.documents import Document

chunk = Document(
    page_content=('| 용 어 풀 이 | 신의성실의 원칙 |\n'
 '| --- | --- |\n'
 '| 신의성실의 원칙이라 함은 계약관계의 당사자는 권리를 행사하거나 의무를 이행 할 때 상대방의 정당한 이익을 배려해야 하고 신뢰를 '
 '저버리지 않도록 행동해야 한다는 원칙을 말합니다.(｢민법｣ 제2조 제1항) | 신의성실의 원칙이라 함은 계약관계의 당사자는 권리를 '
 '행사하거나 의무를 이행 할 때 상대방의 정당한 이익을 배려해야 하고 신뢰를 저버리지 않도록 행동해야 한다는 원칙을 말합니다.(｢민법｣ '
 '제2조 제1항) |\n'
 '| 관 련 법 규 | 민법 제2조(신의성실) 제1항 |'),
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
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
