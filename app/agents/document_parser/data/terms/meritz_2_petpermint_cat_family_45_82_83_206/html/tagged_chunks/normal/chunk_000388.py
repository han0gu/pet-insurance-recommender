from langchain_core.documents import Document

chunk = Document(
    page_content=("것</p><br><p id='38' data-category='paragraph' style='font-size:20px'>\uf000 "
 '제1항에 따라 이 특별약관이 해지된 경우에는 보통약관<br>제35조(해약환급금) 제1항에 따른 해약환급금을 '
 "계약자에게<br>지급합니다.</p><p id='39' data-category='paragraph' "
 "style='font-size:20px'>제18조(보험료의 납입을 연체하여 해지된 계약의 부활(효력<br>회복))</p><br><p "
 "id='40' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000388',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
