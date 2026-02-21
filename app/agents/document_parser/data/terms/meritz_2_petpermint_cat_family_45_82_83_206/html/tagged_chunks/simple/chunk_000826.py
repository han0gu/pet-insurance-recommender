from langchain_core.documents import Document

chunk = Document(
    page_content=("id='73' data-category='paragraph' style='font-size:20px'>내장장기 또는 체강(體腔) 내부를 "
 "직접 볼 수 있게 만든<br>의료기구</p><h1 id='74' style='font-size:20px'>제6조(특별약관의 "
 "소멸)</h1><br><p id='75' data-category='paragraph' style='font-size:16px'>이 "
 '특별약관에서 정한 보상하는 손해가 더 이상 발생할 수<br>없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우<br>회사는「보험료 '
 '및'),
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
 'indexing': {'chunk_id': 'chunk_000826',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
