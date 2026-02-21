from langchain_core.documents import Document

chunk = Document(
    page_content=('- 축산농장에 상시고용된 수의사는 해당 농장의 가축에\n'
 '- 게 투여할 목적으로 동물용 의약품에 대한 처방전을\n'
 '- 발급할 수 있다. 이 경우 상시고용된 수의사의 범위,\n'
 '- 신고방법, 처방전 발급 및 보존 방법, 진료부 작성\n'
 '- 및 보고, 교육, 준수사항 등 그 밖에 필요한 사항은\n'
 '- 농림축산식품부령으로 정한다.\n'
 '\uf000 제1항에도 불구하고 지정된 동물병원에서 진료를 받고\n'
 '「동물병원 보험금 자동청구」절차를 이용한 경우에는 제1\n'
 '항의 서류를 제출한 것으로 간주합니다. 다만, 회사가 보험\n'
 '금 지급을 위해 필요하다고 인정하는 경우 관련 서류를 요'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
