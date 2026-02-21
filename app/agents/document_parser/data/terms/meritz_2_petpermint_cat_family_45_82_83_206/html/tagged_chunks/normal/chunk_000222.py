from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 이 경우<br>에도 회사는 입원일수 20일에 해당하는 보험금을 지급합<br>니다.</p><br><p id='2' "
 "data-category='paragraph' style='font-size:20px'>\uf000 회사가 제1항에 따라 계약을 해지한 "
 '경우 회사는 그 취<br>지를 계약자에게 통지하고 제35조(해약환급금) 제1항에 따<br>른 해약환급금을 지급합니다.</p><h1 '
 "id='3' style='font-size:20px'>제34조(회사의 파산선고와 해지)</h1><br><p id='4'"),
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
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
