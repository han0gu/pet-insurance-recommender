from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 금리연동 형보험은 각 상품별 사업방법서에서 별도로 정한 이율로 계산합니다. ② 제1항에 따라 해지된 특별약관을 '
 '부활(효력회복)하는 경우에는 제11조(계약 전 알릴 의무), 제13조(알릴 의무 위반의 효과), 제15조(사기에 의한 계약), '
 '제20조(제1회 보험 료 및 회사의 보장개시) 및 보통약관 제20조(보험계약의 성립)를 준용합니다. ③ 제1항에서 정한 특별약관의 '
 '부활(효력회복)이 이루어진 경우라도 계약자 또는 피보험 자가 최초계약 청약시 제11조(계약 전 알릴 의무)를 위반한 경우에는 '
 '제13조(알릴 의'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000428',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
