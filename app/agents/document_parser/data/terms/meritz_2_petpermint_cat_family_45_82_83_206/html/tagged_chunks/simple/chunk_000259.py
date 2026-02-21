from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에<br>서 정하지 않은 사항은 "
 "금융소비자보호에 관한 법률, 상법,</p><footer id='60' style='font-size:14px'>81</footer><h1 "
 "id='61' style='font-size:20px'>민법 등 관계 법령을 따릅니다.</h1><p id='62' "
 "data-category='paragraph' style='font-size:20px'>제48조(예금보험에 의한 "
 "지급보장)</p><br><p id='63'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000259',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
