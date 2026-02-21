from langchain_core.documents import Document

chunk = Document(
    page_content=("지급금액</p><br><p id='80' data-category='list' style='font-size:20px'>= [(103만원 "
 "- 3만원)×70%, 100만원] 중 적은금액<br>= 70만원</p><br><p id='81' "
 "data-category='paragraph' style='font-size:20px'>③ 입원 중 MRI,CT 및 내시경처치와 수술을 "
 "동시에 한<br>경우(보상비율 70% 가입)</p><br><p id='82' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000755',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
