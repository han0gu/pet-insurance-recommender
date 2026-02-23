from langchain_core.documents import Document

chunk = Document(
    page_content=('일반적으로<br>도달에 필요한 기간이 지난 때에 계약자 또는 보험수익자에<br>게 도달된 것으로 봅니다.</p><footer '
 "id='96' style='font-size:14px'>56</footer><h1 id='0' "
 "style='font-size:18px'>제13조(보험수익자의 지정)</h1><br><p id='1' "
 "data-category='paragraph' style='font-size:16px'>보험수익자를 지정하지 않은 때에는 보험수익자를 "
 '만기환급<br>금의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자<br>의'),
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
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
