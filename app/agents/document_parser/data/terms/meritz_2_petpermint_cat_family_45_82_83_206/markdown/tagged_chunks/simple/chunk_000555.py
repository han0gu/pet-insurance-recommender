from langchain_core.documents import Document

chunk = Document(
    page_content=('- 수술적 치료) 후 6개월 이상 지난 후에 평가한다.\n'
 '- 5) 신경학적 검사상 나타난 저린감이나 방사통 등 신경자극\n'
 '- 증상의 원인으로 CT, MRI 등 영상검사에서 추간판탈출증\n'
 '- 이 확인된 경우를 추간판탈출증으로 진단하며, 수술 여\n'
 '- 부에 관계없이 운동장해 및 기형장해로 평가하지 않는\n'
 '- 다.\n'
 '- 6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를\n'
 '- 말한다.\n'
 '- 가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 4개 이\n'
 '- 상의 척추체(척추뼈 몸통)를 유합(아물어 붙음)\n'
 '- 또는 고정한 상태'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000555',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
