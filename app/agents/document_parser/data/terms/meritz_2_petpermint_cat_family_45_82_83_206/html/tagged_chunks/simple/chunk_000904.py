from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 해지권을 행사하는 경우 위 표의 ‘청구<br>일’은 보험사의 해지 의사표시(서면, 전자우<br>편, 휴대전화 문자메시지 또는 '
 "이에 준하는 전<br>자적 의사표시 포함)가 보험계약자 또는 그의<br>대리인에게 도달한 날로 봅니다.</p><footer id='31' "
 "style='font-size:14px'>175</footer><p id='32' data-category='paragraph' "
 "style='font-size:18px'>【별표2】</p><h1 id='33'"),
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
 'indexing': {'chunk_id': 'chunk_000904',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
