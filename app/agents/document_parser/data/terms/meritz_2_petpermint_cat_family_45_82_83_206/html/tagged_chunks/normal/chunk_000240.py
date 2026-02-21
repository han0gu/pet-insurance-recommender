from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 회사와 계약자가 합의하여<br>관할법원을 달리 정할 수 있습니다.</p><h1 id='28' "
 "style='font-size:18px'>제41조(소멸시효)</h1><br><p id='29' "
 "data-category='paragraph' style='font-size:16px'>보험금청구권, 만기환급금청구권, 보험료반환청구권, "
 '해약<br>환급금청구권 및 계약자적립액 반환청구권은 3년간 행사하<br>지 않으면 소멸시효가 완성됩니다.</p><br><h1 '
 "id='30'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000240',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
