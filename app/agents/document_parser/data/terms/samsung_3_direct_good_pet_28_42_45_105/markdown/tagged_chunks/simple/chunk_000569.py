from langchain_core.documents import Document

chunk = Document(
    page_content=('유가 발생한 경우 보험료 납입을 면제하여 드리지 않습니다.# 제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))회사는 '
 '이 특별약관의 부활(효력회복) 청약을 받은 경우에는 계약의 부활(효력회복)을 승낙한 경우에 한하여 보험계약 「보험료의 납입을 연체하여 '
 '해지된 계약의 부활(효력회복)」에 따라 이 특별약관의 부활(효력회복)을 취급합니다.# 제 4조 (준용규정)이 특별약관에 정하지 않은 '
 '사항에 대하여는 보험계약을 따릅니다.# 【 붙임1】 특정신체부위 분류표| 구분 | 특 정 신 체 부 위 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
