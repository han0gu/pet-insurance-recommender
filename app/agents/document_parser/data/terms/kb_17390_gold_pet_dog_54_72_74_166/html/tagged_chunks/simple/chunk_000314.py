from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀 이</td><td>신의성실의 원칙</td></tr><tr><td colspan="2">신의성실의 원칙이라 함은 계약관계의 당사자는 '
 '권리를 행사하거나 의무를 이행 할 때 상대방의 정당한 이익을 배려해야 하고 신뢰를 저버리지 않도록 행동해야 한다는 원칙을 '
 '말합니다.(｢민법｣ 제2조 제1항)</td></tr><tr><td>관 련 법 규</td><td>민법 제2조(신의성실) '
 '제1항</td></tr><tr><td colspan="2">① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야'),
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
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
