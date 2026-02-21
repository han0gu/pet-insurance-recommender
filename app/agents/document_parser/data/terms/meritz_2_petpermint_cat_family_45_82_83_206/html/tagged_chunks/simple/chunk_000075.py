from langchain_core.documents import Document

chunk = Document(
    page_content=("2명 이상인 경우 】</h1><br><p id='7' data-category='paragraph' "
 "style='font-size:16px'>계약자가 2명 이상인 경우, 계약 전 알릴 의무, 보험료<br>납입의무 등 보험계약에 따른 "
 "계약자의 의무를 연대로<br>합니다.</p><h1 id='8' style='font-size:18px'>【연대】</h1><br><p "
 "id='9' data-category='paragraph' style='font-size:16px'>2인 이상이 공동으로 책임지는 것을 "
 '뜻하며, 각자가 채<br>무의'),
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
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
