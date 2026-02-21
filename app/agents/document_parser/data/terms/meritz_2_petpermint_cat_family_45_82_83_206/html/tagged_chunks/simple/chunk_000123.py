from langchain_core.documents import Document

chunk = Document(
    page_content=('않은 경우<br>를 말합니다.<br>\uf000 제30조(보험료의 납입을 연체하여 해지된 계약의 부활<br>(효력회복))에서 정한 계약의 '
 "부활이 이루어진 경우 부활을<br>청약한 날을 제5항의 청약일로 하여 적용합니다.</p><h1 id='72' "
 "style='font-size:20px'>제20조(청약의 철회)</h1><br><p id='73' "
 "data-category='paragraph' style='font-size:20px'>\uf000 일반금융소비자인 계약자는 보험증권을 "
 '받은 날부터 15<br>일 이내에 그 청약을 철회할 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
