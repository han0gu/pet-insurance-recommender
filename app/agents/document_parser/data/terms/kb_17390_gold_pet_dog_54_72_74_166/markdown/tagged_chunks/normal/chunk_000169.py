from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에서 정한 해지계약의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피\n'
 '- 보험자가 최초계약 청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약\n'
 '- 포함) 제14조(계약 전 알릴 의무)를 위반한 경우에는 제16조(알릴 의무 위반의 효\n'
 '- 특별\n'
 '# 과)가 적용됩니다.| 용 어 풀 이 | 부활 | 약 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
