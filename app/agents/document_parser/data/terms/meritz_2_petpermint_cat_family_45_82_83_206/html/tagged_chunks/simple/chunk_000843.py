from langchain_core.documents import Document

chunk = Document(
    page_content=('승낙한 경우에 한하여 보<br>통약관 제30조(보험료의 납입을 연체하여 해지된 계약의 부<br>활(효력회복))를 준용합니다.</p><p '
 "id='7' data-category='paragraph' style='font-size:18px'>제4조(준용규정)</p><br><p "
 "id='8' data-category='paragraph' style='font-size:18px'>이 특별약관에 정하지 않은 사항은 "
 "보통약관 및 해당 특별<br>약관을 따릅니다.</p><footer id='9'"),
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
 'indexing': {'chunk_id': 'chunk_000843',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
