from langchain_core.documents import Document

chunk = Document(
    page_content=('력회복))\uf000 회사는 계약자의 해약환급금 청구권에 대한 강제집행,\n'
 '담보권실행, 국세 및 지방세 체납처분절차에 따라 계약이\n'
 '해지된 경우 해지 당시의 보험수익자가 계약자의 동의를 얻\n'
 '어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에\n'
 '지급하고 제23조(계약내용의 변경 등) 제1항의 절차에 따라\n'
 '계약자 명의를 보험수익자로 변경하여 계약의 특별부활(효\n'
 '력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합\n'
 '니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000111',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
