from langchain_core.documents import Document

chunk = Document(
    page_content=("id='64' data-category='paragraph' style='font-size:16px'>이 특별약관에서 정하지 않은 "
 '사항은「반려동물 비용손해<br>관련 특별약관 일반조항」을 따르고,「반려동물 비용손해<br>관련 특별약관 일반조항」에서 정하지 않은 사항은 '
 "보통약<br>관을 따릅니다.</p><footer id='65' style='font-size:14px'>113</footer><p "
 "id='66' data-category='paragraph' style='font-size:20px'>2"),
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
 'indexing': {'chunk_id': 'chunk_000471',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
