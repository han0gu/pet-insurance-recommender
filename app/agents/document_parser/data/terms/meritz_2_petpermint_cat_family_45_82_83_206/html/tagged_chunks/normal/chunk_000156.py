from langchain_core.documents import Document

chunk = Document(
    page_content=('. 전자문서에 전자서명을 한 후에 그 전자서명을 한 사<br>람이 보험계약에 동의한 본인임을 확인할 수 있도록<br>지문정보를 이용하는 '
 '등 법무부장관이 고시하는 요건<br>을 갖추어 작성될 것<br>4'),
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
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
