from langchain_core.documents import Document

chunk = Document(
    page_content=("합니다.</p><br><p id='23' data-category='paragraph' style='font-size:16px'>① "
 '보험증권에 기재된 직업 또는 직무의 변경<br>1) 현재의 직업 또는 직무가 변경된 경우<br>2) 직업이 없는 자가 취직한 '
 "경우<br>3) 현재의 직업을 그만둔 경우</p><footer id='24' "
 "style='font-size:14px'>58</footer><h1 id='25' "
 "style='font-size:20px'>【직업】</h1><br><p id='26'"),
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
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
