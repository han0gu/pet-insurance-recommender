from langchain_core.documents import Document

chunk = Document(
    page_content=('만기환급<br>금의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자<br>의 법정상속인, 이 외의 보험금은 피보험자로 '
 "합니다.</p><br><h1 id='2' style='font-size:16px'>【법정상속인】</h1><br><p id='3' "
 "data-category='paragraph' style='font-size:18px'>법정상속인이라 함은 피상속인의 사망에 의하여 "
 "민법의<br>규정에 의한 상속순위에 따라 상속받는 자를 말합니다.</p><h1 id='4' "
 "style='font-size:18px'>제14조(대표자의"),
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
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
