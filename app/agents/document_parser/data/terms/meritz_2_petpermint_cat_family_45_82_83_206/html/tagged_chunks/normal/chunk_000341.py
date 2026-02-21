from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>【보험가입금액 제한】</h1><br><p id='85' "
 "data-category='paragraph' style='font-size:20px'>반려동물이 가입을 할 수 있는 최대 보험가입금액을 "
 "제<br>한하는 방법을 말합니다.</p><h1 id='86' style='font-size:20px'>【일부보장 "
 "제외(부담보)】</h1><br><p id='87' data-category='paragraph' "
 "style='font-size:20px'>일반적인 경우보다 위험이 높은 반려동물이 가입하기"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000341',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
