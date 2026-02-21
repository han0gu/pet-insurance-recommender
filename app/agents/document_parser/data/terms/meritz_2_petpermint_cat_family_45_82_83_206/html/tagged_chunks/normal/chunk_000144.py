from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 이미 납입<br>한 보험료를 계약자에게 돌려 드리며, 보험료를 받은 기간<br>에 대하여 보험계약대출이율을 연단위 복리로 계산한 '
 "금액<br>을 더하여 지급합니다.</p><br><p id='102' data-category='list'></p><h1 id='103' "
 "style='font-size:20px'>【보험계약대출이율】</h1><br><p id='104' "
 "data-category='paragraph' style='font-size:20px'>계약자는 해당 계약의 해약환급금 범위내에서 회사가 "
 '정<br>한 방법에 따라'),
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
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
