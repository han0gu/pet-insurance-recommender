from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 이 특별약관에서의 피보험자 및 반려동물의 나이는 만나<br>이를 기준으로 "
 '합니다.<br>\uf000 제1항의 만나이는 계약일 현재 피보험자 및 반려동물의<br>실제 만나이를 기준으로 하며, 이후 매년 '
 '계약해당일에 나<br>이가 증가하는 것으로 합니다.<br>\uf000 반려동물의 나이 및 품종에 관한 청약서상 기재사항이<br>사실과 '
 '다른 경우에는 정정된 나이 및 품종에 해당하는 보<br>험금 및 보험료로 변경합니다'),
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
 'indexing': {'chunk_id': 'chunk_000359',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
