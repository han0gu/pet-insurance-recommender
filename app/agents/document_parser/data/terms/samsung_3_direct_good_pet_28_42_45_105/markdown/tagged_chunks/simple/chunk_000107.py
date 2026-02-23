from langchain_core.documents import Document

chunk = Document(
    page_content=('- 거절할 때에는 거절 사유를 함께 통지하여야 합니다.\n'
 '- ③ 계약자는 회사가 정당한 사유 없이 제1항의 요구를 따르지 않는 경우 해당 계약을 해\n'
 '- 지할 수 있습니다.\n'
 '- ④ 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제33조(해약환급금) 제5항에 따른\n'
 '- 해약환급금을 계약자에게 지급합니다.\n'
 '- ⑤ 계약자는 제1항에 따른 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에\n'
 '- 따라 법률상의 권리를 행사할 수 있습니다.'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
