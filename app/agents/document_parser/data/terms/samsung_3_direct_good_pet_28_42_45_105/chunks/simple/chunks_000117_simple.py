from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이때 회사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다. ③ 제1항에서 정한 계약의 '
 '부활(효력회복)이 이루어진 경우라도 계약자 또는 피보험자가 최초계약 청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) '
 '제13조 (계약 전 알릴 의무)를 위반한 경우에는 제15조(알릴 의무 위반의 효과)가 적용됩니 다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
