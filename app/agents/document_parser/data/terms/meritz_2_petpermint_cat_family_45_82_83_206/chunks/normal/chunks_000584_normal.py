from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제1항에서 지정한 특정질병의 합병증으로 인하여 진단 확정된 특정질병 이외의 질병으로 계약에서 정한 보험 금의 지급사유가 발생한 경우 '
 '② 상해를 직접적인 원인으로 하여 보험금의 지급사유가 발생한 경우 ③ 제1항에서 지정한 특정질병으로 인하여 사망하여 보험 금의 지급사유가 '
 '발생한 경우'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 167},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000584',
              'chunk_char_len': 159,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
