from langchain_core.documents import Document

chunk = Document(
    page_content=('- 력회복)을 거절하지 않습니다.\n'
 '- ③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약 청약시(2회\n'
 '- 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제12조(계약 전 알릴의무)를 위반한 경우에\n'
 '- 는 제26조(계약의 해지) 제3항이 적용됩니다.\n'
 '# 제25조[강제집행 등으로 인한 해지계약의 특별부활(효력회복)]① 타인을 위한 계약의 경우 제30조(보험료의 환급)에 따른 계약자의 '
 '환급금 청구권에 대한 강제집행,'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
