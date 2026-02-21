from langchain_core.documents import Document

chunk = Document(
    page_content=('- (계약 전 알릴 의무)를 위반한 경우에는 제15조(알릴 의무 위반의 효과)가 적용됩니\n'
 '- 다.\n'
 '# 제29조 (강제집행 등으로 인하여 해지된 계약의 특별부활(효력회복))① 회사는 계약자의 해약환급금 청구권에 대한 강제집행, '
 '담보권실행, 국세 및 지방세 체'),
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
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
