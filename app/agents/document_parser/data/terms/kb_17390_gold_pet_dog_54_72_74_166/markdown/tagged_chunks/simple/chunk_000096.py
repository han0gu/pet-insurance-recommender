from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 경우를 포함합니다.\n'
 '- 1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우\n'
 '2. 부담보가 지정된 질병 또는 증상이 악화되지 않고 유지된 경우\uf000 제5항의 "청약일로부터 5년이 지나는 동안"이라 함은 '
 '제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에서 정한 계약의 해지가 발생하지 않# 은 경우를 '
 '말합니다.\uf000 제29조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에서 정한 계약의 부활이 이루어진 경우 부활을 '
 '청약한 날을 제5항의 청약일로 하여 적용합니다.| 용 어 풀 | 이 |'),
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
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
