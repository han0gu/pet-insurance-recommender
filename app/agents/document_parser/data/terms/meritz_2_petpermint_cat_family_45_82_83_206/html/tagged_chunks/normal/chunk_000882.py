from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="21">6</td><td rowspan="21">소화기 질환</td><td>ABB002</td><td>소화관 '
 '림프종</td></tr><tr><td>ABA003</td><td>기타 소화기 계통의 양성 '
 '신생물</td></tr><tr><td>ABB003</td><td>기타 소화기 계통의 악성 '
 '신생물</td></tr><tr><td>ABC003</td><td>기타 소화기 계통의 신생물(양성 또는 악성이'),
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
 'indexing': {'chunk_id': 'chunk_000882',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
